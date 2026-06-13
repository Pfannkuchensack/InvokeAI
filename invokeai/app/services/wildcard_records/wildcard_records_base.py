from abc import ABC, abstractmethod

from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardRecordDTO,
    WildcardWithoutId,
)


class WildcardRecordsStorageBase(ABC):
    """Base class for wildcard storage services."""

    @abstractmethod
    def get(self, wildcard_id: str) -> WildcardRecordDTO:
        """Get a wildcard by id. Authorization is the caller's responsibility."""
        pass

    @abstractmethod
    def create(self, wildcard: WildcardWithoutId, user_id: str) -> WildcardRecordDTO:
        """Creates a wildcard owned by user_id. Raises WildcardNameConflictError on duplicate name."""
        pass

    @abstractmethod
    def create_many(self, wildcards: list[WildcardWithoutId], user_id: str) -> list[WildcardRecordDTO]:
        """Creates many wildcards owned by user_id, overwriting any of the user's existing same-named wildcards."""
        pass

    @abstractmethod
    def update(self, wildcard_id: str, changes: WildcardChanges) -> WildcardRecordDTO:
        """Updates a wildcard. Authorization is the caller's responsibility."""
        pass

    @abstractmethod
    def delete(self, wildcard_id: str) -> None:
        """Deletes a wildcard. Authorization is the caller's responsibility."""
        pass

    @abstractmethod
    def get_many(self, user_id: str | None = None, is_admin: bool = False) -> list[WildcardRecordDTO]:
        """Gets wildcards visible to user_id.

        Visibility rules:
        - is_admin=True: all wildcards.
        - Else: wildcards owned by user_id, plus any public wildcard.
        - If user_id is None and is_admin is False: only public wildcards.
        """
        pass
