import json
from pathlib import PurePosixPath
from typing import Any, Optional

import pydantic
from fastapi import UploadFile
from pydantic import BaseModel, Field, TypeAdapter, field_validator

# The dynamicprompts wildcard wrap. A wildcard name must not contain it, or it would break parsing.
WILDCARD_WRAP = "__"
MAX_WILDCARD_NAME_LENGTH = 256


class WildcardNotFoundError(Exception):
    """Raised when a wildcard is not found."""


class WildcardNameConflictError(Exception):
    """Raised when creating/renaming a wildcard to a name the user already has."""


def _validate_wildcard_name(name: str) -> str:
    """Normalize and validate a wildcard name.

    Names are referenced in prompts as `__name__`. Subdirectory-style names (`group/name`) are
    allowed, but the wildcard wrap, whitespace-only names, and path traversal are not.
    """
    name = name.strip()
    if not name:
        raise ValueError("Wildcard name must not be empty")
    if len(name) > MAX_WILDCARD_NAME_LENGTH:
        raise ValueError(f"Wildcard name must be at most {MAX_WILDCARD_NAME_LENGTH} characters")
    if WILDCARD_WRAP in name:
        raise ValueError(f"Wildcard name must not contain '{WILDCARD_WRAP}'")
    if any(ch in name for ch in ("\\", "\n", "\r", "\t")):
        raise ValueError("Wildcard name contains invalid characters")
    # Disallow path traversal / absolute paths for subdirectory-style names.
    parts = PurePosixPath(name).parts
    if name.startswith("/") or any(part in ("..", ".") for part in parts):
        raise ValueError("Wildcard name contains invalid path segments")
    return name


class WildcardWithoutId(BaseModel):
    name: str = Field(description="The name of the wildcard, referenced in prompts as `__name__`.")
    values: list[str] = Field(description="The list of values the wildcard expands to.")
    is_public: bool = Field(default=False, description="Whether the wildcard is visible to other users.")

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        return _validate_wildcard_name(v)

    @field_validator("values")
    @classmethod
    def _clean_values(cls, v: list[str]) -> list[str]:
        # Drop blank lines and surrounding whitespace; these are never useful wildcard values.
        return [line.strip() for line in v if line.strip()]


class WildcardRecordDTO(WildcardWithoutId):
    id: str = Field(description="The wildcard ID.")
    user_id: str = Field(description="The user who owns this wildcard.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WildcardRecordDTO":
        data["values"] = json.loads(data.get("values", "[]"))
        data["is_public"] = bool(data.get("is_public", False))
        return WildcardRecordDTOValidator.validate_python(data)


WildcardRecordDTOValidator = TypeAdapter(WildcardRecordDTO)


class WildcardChanges(BaseModel, extra="forbid"):
    name: Optional[str] = Field(default=None, description="The wildcard's new name.")
    values: Optional[list[str]] = Field(default=None, description="The updated list of values.")
    is_public: Optional[bool] = Field(default=None, description="Whether the wildcard is visible to other users.")

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: Optional[str]) -> Optional[str]:
        return _validate_wildcard_name(v) if v is not None else None

    @field_validator("values")
    @classmethod
    def _clean_values(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        return [line.strip() for line in v if line.strip()] if v is not None else None


class UnsupportedFileTypeError(ValueError):
    """Raised when an unsupported file type is encountered during import."""


class InvalidWildcardImportDataError(ValueError):
    """Raised when invalid wildcard import data is encountered."""


def wildcards_to_root_map_dict(
    records: list[WildcardRecordDTO],
    owner_user_id: str | None = None,
) -> dict[str, list[str]]:
    """Convert wildcard records into a `{name: values}` mapping for the dynamicprompts WildcardManager.

    If two records share a name (e.g. a user's own wildcard and a public one), the owner's wildcard
    takes precedence.
    """
    result: dict[str, list[str]] = {}
    # Process non-owned first so owned entries (processed last) overwrite on name collisions.
    for record in sorted(records, key=lambda r: r.user_id == owner_user_id):
        result[record.name] = record.values
    return result


async def parse_wildcards_from_file(file: UploadFile) -> list[WildcardWithoutId]:
    """Parse wildcards from a single uploaded file.

    Supported formats:
    - `.txt`: one collection; name = filename (without extension), values = non-empty lines.
    - `.json`: either an object `{"name": ["value", ...], ...}` or a list of
      `{"name": "...", "values": ["..."]}` objects.

    Raises:
        UnsupportedFileTypeError: If the file type is not supported.
        InvalidWildcardImportDataError: If the data in the file is invalid.
    """
    filename = file.filename or ""
    suffix = PurePosixPath(filename).suffix.lower()

    try:
        contents = (await file.read()).decode("utf-8")
    except UnicodeDecodeError as e:
        raise InvalidWildcardImportDataError("File is not valid UTF-8 text") from e
    finally:
        file.file.close()

    if suffix == ".txt":
        name = PurePosixPath(filename).stem
        try:
            return [WildcardWithoutId(name=name, values=contents.splitlines())]
        except pydantic.ValidationError as e:
            raise InvalidWildcardImportDataError(f"Invalid wildcard file '{filename}': {e}") from e

    if suffix == ".json":
        try:
            data = json.loads(contents)
        except json.JSONDecodeError as e:
            raise InvalidWildcardImportDataError(f"Invalid JSON in '{filename}'") from e

        try:
            if isinstance(data, dict):
                return [WildcardWithoutId(name=name, values=list(values)) for name, values in data.items()]
            if isinstance(data, list):
                return [WildcardWithoutId(**entry) for entry in data]
        except (pydantic.ValidationError, TypeError, ValueError) as e:
            raise InvalidWildcardImportDataError(f"Invalid wildcard data in '{filename}': {e}") from e

        raise InvalidWildcardImportDataError(
            f"Invalid JSON structure in '{filename}': expected an object or a list of objects"
        )

    raise UnsupportedFileTypeError(f"Unsupported file type '{suffix or filename}'. Use .txt or .json.")
