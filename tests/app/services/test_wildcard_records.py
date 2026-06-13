from __future__ import annotations

import pytest

from invokeai.app.services.config.config_default import InvokeAIAppConfig
from invokeai.app.services.users.users_common import UserCreateRequest
from invokeai.app.services.users.users_default import UserService
from invokeai.app.services.wildcard_records.wildcard_records_common import (
    WildcardChanges,
    WildcardNameConflictError,
    WildcardNotFoundError,
    WildcardWithoutId,
    wildcards_to_root_map_dict,
)
from invokeai.app.services.wildcard_records.wildcard_records_sqlite import SqliteWildcardRecordsStorage
from invokeai.backend.util.logging import InvokeAILogger
from tests.fixtures.sqlite_database import create_mock_sqlite_database


@pytest.fixture
def storage_and_users() -> tuple[SqliteWildcardRecordsStorage, str, str]:
    cfg = InvokeAIAppConfig(use_memory_db=True, node_cache_size=0)
    db = create_mock_sqlite_database(cfg, InvokeAILogger.get_logger())
    users = UserService(db)
    u1 = users.create(UserCreateRequest(email="u1@t.com", display_name="u1", password="TestPass123")).user_id
    u2 = users.create(UserCreateRequest(email="u2@t.com", display_name="u2", password="TestPass123")).user_id
    return SqliteWildcardRecordsStorage(db=db), u1, u2


def test_create_and_get(storage_and_users) -> None:
    storage, u1, _ = storage_and_users
    created = storage.create(WildcardWithoutId(name="animals", values=["cat", "dog"]), user_id=u1)
    assert created.name == "animals"
    assert created.values == ["cat", "dog"]
    assert created.user_id == u1
    assert created.is_public is False
    assert storage.get(created.id).id == created.id


def test_get_missing_raises(storage_and_users) -> None:
    storage, _, _ = storage_and_users
    with pytest.raises(WildcardNotFoundError):
        storage.get("does-not-exist")


def test_duplicate_name_per_user_conflicts(storage_and_users) -> None:
    storage, u1, u2 = storage_and_users
    storage.create(WildcardWithoutId(name="animals", values=["cat"]), user_id=u1)
    with pytest.raises(WildcardNameConflictError):
        storage.create(WildcardWithoutId(name="animals", values=["dog"]), user_id=u1)
    # A different user may reuse the same name.
    assert storage.create(WildcardWithoutId(name="animals", values=["dog"]), user_id=u2).name == "animals"


def test_get_many_visibility(storage_and_users) -> None:
    storage, u1, u2 = storage_and_users
    storage.create(WildcardWithoutId(name="own", values=["a"]), user_id=u1)
    storage.create(WildcardWithoutId(name="shared", values=["b"], is_public=True), user_id=u2)
    storage.create(WildcardWithoutId(name="private", values=["c"], is_public=False), user_id=u2)

    visible = {w.name for w in storage.get_many(user_id=u1, is_admin=False)}
    assert visible == {"own", "shared"}  # own + public, never another user's private

    all_for_admin = {w.name for w in storage.get_many(user_id=u1, is_admin=True)}
    assert all_for_admin == {"own", "shared", "private"}


def test_update_rename_values_and_visibility(storage_and_users) -> None:
    storage, u1, _ = storage_and_users
    created = storage.create(WildcardWithoutId(name="animals", values=["cat"]), user_id=u1)
    updated = storage.update(
        created.id, WildcardChanges(name="critters", values=["cat", "fox"], is_public=True)
    )
    assert updated.name == "critters"
    assert updated.values == ["cat", "fox"]
    assert updated.is_public is True


def test_update_rename_conflict(storage_and_users) -> None:
    storage, u1, _ = storage_and_users
    storage.create(WildcardWithoutId(name="a", values=["x"]), user_id=u1)
    b = storage.create(WildcardWithoutId(name="b", values=["y"]), user_id=u1)
    with pytest.raises(WildcardNameConflictError):
        storage.update(b.id, WildcardChanges(name="a"))


def test_create_many_upserts_same_name(storage_and_users) -> None:
    storage, u1, _ = storage_and_users
    storage.create(WildcardWithoutId(name="animals", values=["cat"]), user_id=u1)
    storage.create_many(
        [
            WildcardWithoutId(name="animals", values=["dog", "fox"]),  # overwrites existing
            WildcardWithoutId(name="new", values=["a", "b"]),
        ],
        user_id=u1,
    )
    by_name = {w.name: w.values for w in storage.get_many(user_id=u1, is_admin=False)}
    assert by_name == {"animals": ["dog", "fox"], "new": ["a", "b"]}


def test_delete(storage_and_users) -> None:
    storage, u1, _ = storage_and_users
    created = storage.create(WildcardWithoutId(name="animals", values=["cat"]), user_id=u1)
    storage.delete(created.id)
    with pytest.raises(WildcardNotFoundError):
        storage.get(created.id)


def test_wildcards_to_root_map_dict_owner_precedence(storage_and_users) -> None:
    storage, u1, u2 = storage_and_users
    storage.create(WildcardWithoutId(name="colors", values=["own-red"]), user_id=u1)
    storage.create(WildcardWithoutId(name="colors", values=["public-blue"], is_public=True), user_id=u2)
    mapping = wildcards_to_root_map_dict(storage.get_many(user_id=u1, is_admin=False), owner_user_id=u1)
    # The user's own wildcard wins over a public one with the same name.
    assert mapping["colors"] == ["own-red"]
