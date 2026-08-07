"""Record the size variant on existing Qwen3-VL encoder models.

``Qwen3VLEncoder`` configs gained a required ``variant`` field (4B / 8B) when Ideogram 4 became a
second consumer alongside Krea-2. Records written before that carry no ``variant``, and the field
has no default — deliberately, since a default would fold it into the discriminator tag — so
without this migration they would fail validation on load.

Backfilling is unambiguous: the config previously *rejected* anything whose text-tower hidden size
was not 2560, so every Qwen3-VL encoder that could have been installed is a 4B.
"""

import json
import sqlite3
from logging import Logger
from typing import Any

from invokeai.app.services.config import InvokeAIAppConfig
from invokeai.app.services.shared.sqlite_migrator.sqlite_migrator_common import Migration
from invokeai.backend.model_manager.taxonomy import ModelType, Qwen3VLVariantType


class Qwen3VLEncoderVariantCallback:
    def __init__(self, app_config: InvokeAIAppConfig, logger: Logger) -> None:
        self._app_config = app_config
        self._logger = logger

    def __call__(self, cursor: sqlite3.Cursor) -> None:
        cursor.execute("SELECT id, config FROM models;")
        rows = cursor.fetchall()

        migrated = 0
        for model_id, config_json in rows:
            try:
                config_dict: dict[str, Any] = json.loads(config_json)
            except json.JSONDecodeError as e:
                self._logger.error("Invalid config JSON for model %s: %s", model_id, e)
                raise

            if config_dict.get("type") != ModelType.Qwen3VLEncoder.value:
                continue
            if "variant" in config_dict:
                continue

            config_dict["variant"] = Qwen3VLVariantType.Qwen3VL_4B.value
            cursor.execute(
                "UPDATE models SET config = ? WHERE id = ?;",
                (json.dumps(config_dict), model_id),
            )
            migrated += 1

        if migrated:
            self._logger.info(f"Migration complete: {migrated} Qwen3-VL encoder config(s) marked as 4B")
        else:
            self._logger.info("Migration complete: no Qwen3-VL encoder configs needed migration")


def build_migration(app_config: InvokeAIAppConfig, logger: Logger) -> Migration:
    return Migration(
        id="2026_08_07_qwen3_vl_encoder_variant",
        depends_on="migration_33",
        callback=Qwen3VLEncoderVariantCallback(app_config=app_config, logger=logger),
    )
