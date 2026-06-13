"""Migration 32: Add wildcards table for dynamic prompt wildcards.

This migration adds the `wildcards` table, which stores user-managed dynamic prompt
wildcard collections. Each row is a named collection of string values, owned by a user
and optionally shared publicly (mirroring the ownership/visibility model used for style
presets and workflows). The `values` column holds a JSON array of strings.
"""

import sqlite3

from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration


class Migration32Callback:
    """Migration to add the wildcards table."""

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        self._create_wildcards_table(cursor)

    def _create_wildcards_table(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wildcards (
                id TEXT NOT NULL PRIMARY KEY,
                name TEXT NOT NULL,
                "values" TEXT NOT NULL DEFAULT '[]',
                user_id TEXT NOT NULL DEFAULT 'system',
                is_public BOOLEAN NOT NULL DEFAULT FALSE,
                created_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                updated_at DATETIME NOT NULL DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
        """)

        # A wildcard name must be unique per user — the name is the lookup key in prompts (`__name__`).
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_wildcards_user_id_name ON wildcards(user_id, name);"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wildcards_user_id ON wildcards(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wildcards_is_public ON wildcards(is_public);")

        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS tg_wildcards_updated_at
            AFTER UPDATE ON wildcards FOR EACH ROW
            BEGIN
                UPDATE wildcards SET updated_at = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
                WHERE id = old.id;
            END;
        """)


def build_migration_32() -> Migration:
    """Builds the migration object for migrating from version 31 to version 32.

    This migration adds the wildcards table for user-managed dynamic prompt wildcards.
    """
    return Migration(
        from_version=31,
        to_version=32,
        callback=Migration32Callback(),
    )
