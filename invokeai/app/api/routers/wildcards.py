import traceback

from fastapi import APIRouter, Body, File, HTTPException, Path, UploadFile

from invokeai.app.api.auth_dependencies import CurrentUserOrDefault
from invokeai.app.api.dependencies import ApiDependencies
from invokeai.app.services.auth.token_service import TokenData
from invokeai.app.services.wildcard_records.wildcard_records_common import (
    InvalidWildcardImportDataError,
    UnsupportedFileTypeError,
    WildcardChanges,
    WildcardNameConflictError,
    WildcardNotFoundError,
    WildcardRecordDTO,
    WildcardWithoutId,
    parse_wildcards_from_file,
)

wildcards_router = APIRouter(prefix="/v1/wildcards", tags=["wildcards"])


def _assert_wildcard_read(record: WildcardRecordDTO, current_user: TokenData) -> None:
    """Allow read access if admin, owner, or public wildcard."""
    if current_user.is_admin:
        return
    if record.is_public:
        return
    if record.user_id == current_user.user_id:
        return
    raise HTTPException(status_code=403, detail="Not authorized to access this wildcard")


def _assert_wildcard_write(record: WildcardRecordDTO, current_user: TokenData) -> None:
    """Allow write access only for admin or owner."""
    if current_user.is_admin:
        return
    if record.user_id == current_user.user_id:
        return
    raise HTTPException(status_code=403, detail="Not authorized to modify this wildcard")


def _load_record_or_404(wildcard_id: str) -> WildcardRecordDTO:
    try:
        return ApiDependencies.invoker.services.wildcard_records.get(wildcard_id)
    except WildcardNotFoundError:
        raise HTTPException(status_code=404, detail="Wildcard not found")


@wildcards_router.get(
    "/",
    operation_id="list_wildcards",
    responses={200: {"model": list[WildcardRecordDTO]}},
)
async def list_wildcards(current_user: CurrentUserOrDefault) -> list[WildcardRecordDTO]:
    """Gets the wildcards visible to the current user (own + public)."""
    return ApiDependencies.invoker.services.wildcard_records.get_many(
        user_id=current_user.user_id,
        is_admin=current_user.is_admin,
    )


@wildcards_router.post(
    "/",
    operation_id="create_wildcard",
    responses={200: {"model": WildcardRecordDTO}},
)
async def create_wildcard(
    current_user: CurrentUserOrDefault,
    wildcard: WildcardWithoutId = Body(description="The wildcard to create"),
) -> WildcardRecordDTO:
    """Creates a wildcard owned by the current user."""
    try:
        return ApiDependencies.invoker.services.wildcard_records.create(
            wildcard=wildcard, user_id=current_user.user_id
        )
    except WildcardNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@wildcards_router.get(
    "/i/{wildcard_id}",
    operation_id="get_wildcard",
    responses={200: {"model": WildcardRecordDTO}},
)
async def get_wildcard(
    current_user: CurrentUserOrDefault,
    wildcard_id: str = Path(description="The wildcard to get"),
) -> WildcardRecordDTO:
    """Gets a wildcard."""
    record = _load_record_or_404(wildcard_id)
    _assert_wildcard_read(record, current_user)
    return record


@wildcards_router.patch(
    "/i/{wildcard_id}",
    operation_id="update_wildcard",
    responses={200: {"model": WildcardRecordDTO}},
)
async def update_wildcard(
    current_user: CurrentUserOrDefault,
    wildcard_id: str = Path(description="The id of the wildcard to update"),
    changes: WildcardChanges = Body(description="The changes to apply to the wildcard"),
) -> WildcardRecordDTO:
    """Updates a wildcard."""
    record = _load_record_or_404(wildcard_id)
    _assert_wildcard_write(record, current_user)
    try:
        return ApiDependencies.invoker.services.wildcard_records.update(wildcard_id=wildcard_id, changes=changes)
    except WildcardNameConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))


@wildcards_router.delete(
    "/i/{wildcard_id}",
    operation_id="delete_wildcard",
)
async def delete_wildcard(
    current_user: CurrentUserOrDefault,
    wildcard_id: str = Path(description="The wildcard to delete"),
) -> None:
    """Deletes a wildcard."""
    record = _load_record_or_404(wildcard_id)
    _assert_wildcard_write(record, current_user)
    ApiDependencies.invoker.services.wildcard_records.delete(wildcard_id)


@wildcards_router.post(
    "/import",
    operation_id="import_wildcards",
    responses={200: {"model": list[WildcardRecordDTO]}},
)
async def import_wildcards(
    current_user: CurrentUserOrDefault,
    files: list[UploadFile] = File(description="The wildcard files to import (.txt or .json)"),
) -> list[WildcardRecordDTO]:
    """Imports wildcards from one or more files, owned by the current user.

    A `.txt` file becomes a single wildcard (name = filename, values = lines). A `.json` file may be
    an object `{name: [values]}` or a list of `{name, values}` objects. Importing overwrites any of
    the user's existing same-named wildcards.
    """
    wildcards: list[WildcardWithoutId] = []
    try:
        for file in files:
            wildcards.extend(await parse_wildcards_from_file(file))
    except InvalidWildcardImportDataError as e:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except UnsupportedFileTypeError as e:
        ApiDependencies.invoker.services.logger.error(traceback.format_exc())
        raise HTTPException(status_code=415, detail=str(e))

    return ApiDependencies.invoker.services.wildcard_records.create_many(wildcards, user_id=current_user.user_id)
