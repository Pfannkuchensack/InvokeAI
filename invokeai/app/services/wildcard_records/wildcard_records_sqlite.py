import json
import sqlite3

from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.sqlite.sqlite_database import SqliteDatabase
from invokeai.app.services.wildcard_records.wildcard_records_base import WildcardRecordsStorageBase
from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardNameConflictError,
    WildcardNotFoundError,
    WildcardRecordDTO,
    WildcardWithoutId,
)
from invokeai.app.util.misc import uuid_string


class SqliteWildcardRecordsStorage(WildcardRecordsStorageBase):
    def __init__(self, db: SqliteDatabase) -> None:
        super().__init__()
        self._db = db

    def start(self, invoker: Invoker) -> None:
        self._invoker = invoker

    def get(self, wildcard_id: str) -> WildcardRecordDTO:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                SELECT * FROM wildcards WHERE id = ?;
                """,
                (wildcard_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise WildcardNotFoundError(f"Wildcard with id {wildcard_id} not found")
        return WildcardRecordDTO.from_dict(dict(row))

    def create(self, wildcard: WildcardWithoutId, user_id: str) -> WildcardRecordDTO:
        wildcard_id = uuid_string()
        try:
            with self._db.transaction() as cursor:
                cursor.execute(
                    """--sql
                    INSERT INTO wildcards (id, name, "values", user_id, is_public)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        wildcard_id,
                        wildcard.name,
                        json.dumps(wildcard.values),
                        user_id,
                        1 if wildcard.is_public else 0,
                    ),
                )
        except sqlite3.IntegrityError as e:
            if "UNIQUE" in str(e):
                raise WildcardNameConflictError(f"A wildcard named '{wildcard.name}' already exists") from e
            raise
        return self.get(wildcard_id)

    def create_many(self, wildcards: list[WildcardWithoutId], user_id: str) -> list[WildcardRecordDTO]:
        ids: list[str] = []
        with self._db.transaction() as cursor:
            for wildcard in wildcards:
                wildcard_id = uuid_string()
                # Import overwrites any of the user's existing same-named wildcard (upsert on user_id+name).
                cursor.execute(
                    """--sql
                    INSERT INTO wildcards (id, name, "values", user_id, is_public)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, name) DO UPDATE SET
                        "values" = excluded."values",
                        is_public = excluded.is_public;
                    """,
                    (
                        wildcard_id,
                        wildcard.name,
                        json.dumps(wildcard.values),
                        user_id,
                        1 if wildcard.is_public else 0,
                    ),
                )
            # Fetch the resulting rows by (user_id, name) so we return the actual stored ids.
            names = [w.name for w in wildcards]
            placeholders = ",".join("?" for _ in names)
            cursor.execute(
                f"""--sql
                SELECT * FROM wildcards WHERE user_id = ? AND name IN ({placeholders});
                """,
                (user_id, *names),
            )
            rows = cursor.fetchall()
            ids = [row["id"] for row in rows]
        return [self.get(wildcard_id) for wildcard_id in ids]

    def update(self, wildcard_id: str, changes: WildcardChanges) -> WildcardRecordDTO:
        try:
            with self._db.transaction() as cursor:
                if changes.name is not None:
                    cursor.execute(
                        """--sql
                        UPDATE wildcards SET name = ? WHERE id = ?;
                        """,
                        (changes.name, wildcard_id),
                    )
                if changes.values is not None:
                    cursor.execute(
                        """--sql
                        UPDATE wildcards SET "values" = ? WHERE id = ?;
                        """,
                        (json.dumps(changes.values), wildcard_id),
                    )
                if changes.is_public is not None:
                    cursor.execute(
                        """--sql
                        UPDATE wildcards SET is_public = ? WHERE id = ?;
                        """,
                        (1 if changes.is_public else 0, wildcard_id),
                    )
        except sqlite3.IntegrityError as e:
            if "UNIQUE" in str(e):
                raise WildcardNameConflictError(f"A wildcard named '{changes.name}' already exists") from e
            raise
        return self.get(wildcard_id)

    def delete(self, wildcard_id: str) -> None:
        with self._db.transaction() as cursor:
            cursor.execute(
                """--sql
                DELETE FROM wildcards WHERE id = ?;
                """,
                (wildcard_id,),
            )

    def get_many(self, user_id: str | None = None, is_admin: bool = False) -> list[WildcardRecordDTO]:
        clauses: list[str] = []
        params: list[object] = []

        if not is_admin:
            # Visible to non-admin: own + public.
            visibility = "(is_public = 1"
            if user_id is not None:
                visibility += " OR user_id = ?"
                params.append(user_id)
            visibility += ")"
            clauses.append(visibility)

        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        query = f'SELECT * FROM wildcards {where}ORDER BY LOWER(name) ASC'

        with self._db.transaction() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [WildcardRecordDTO.from_dict(dict(row)) for row in rows]
