"""Router-level tests for /api/v1/wildcards.

Covers auth gating, CRUD, per-user visibility/authorization, file import, and the
integration with the dynamicprompts endpoint (a created wildcard resolves in prompts).
"""

from typing import Any

from fastapi import status
from fastapi.testclient import TestClient

from invokeai.app.services.invoker import Invoker


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_wildcards_require_auth(enable_multiuser: Any, client: TestClient):
    r = client.get("/api/v1/wildcards/")
    assert r.status_code == status.HTTP_401_UNAUTHORIZED


def test_create_and_list_wildcard(client: TestClient, user1_token: str):
    r = client.post(
        "/api/v1/wildcards/",
        json={"name": "animals", "values": ["cat", "dog"]},
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_200_OK, r.text
    created = r.json()
    assert created["name"] == "animals"
    assert created["values"] == ["cat", "dog"]
    assert created["is_public"] is False

    r = client.get("/api/v1/wildcards/", headers=_auth(user1_token))
    assert r.status_code == status.HTTP_200_OK
    assert [w["name"] for w in r.json()] == ["animals"]


def test_create_blank_values_and_name_are_cleaned(client: TestClient, user1_token: str):
    r = client.post(
        "/api/v1/wildcards/",
        json={"name": "  spaced  ", "values": ["a", "  ", "", " b "]},
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_200_OK, r.text
    created = r.json()
    assert created["name"] == "spaced"
    assert created["values"] == ["a", "b"]


def test_create_invalid_name_rejected(client: TestClient, user1_token: str):
    # `__` is the wildcard wrap and must not appear in a name.
    r = client.post(
        "/api/v1/wildcards/",
        json={"name": "bad__name", "values": ["a"]},
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_duplicate_name_conflicts(client: TestClient, user1_token: str):
    body = {"name": "animals", "values": ["cat"]}
    assert client.post("/api/v1/wildcards/", json=body, headers=_auth(user1_token)).status_code == 200
    r = client.post("/api/v1/wildcards/", json=body, headers=_auth(user1_token))
    assert r.status_code == status.HTTP_409_CONFLICT


def test_update_and_delete(client: TestClient, user1_token: str):
    created = client.post(
        "/api/v1/wildcards/", json={"name": "animals", "values": ["cat"]}, headers=_auth(user1_token)
    ).json()
    wid = created["id"]

    r = client.patch(
        f"/api/v1/wildcards/i/{wid}",
        json={"values": ["cat", "dog"], "is_public": True},
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_200_OK, r.text
    assert r.json()["values"] == ["cat", "dog"]
    assert r.json()["is_public"] is True

    assert client.delete(f"/api/v1/wildcards/i/{wid}", headers=_auth(user1_token)).status_code == 200
    assert client.get(f"/api/v1/wildcards/i/{wid}", headers=_auth(user1_token)).status_code == 404


def test_user_cannot_access_others_private_wildcard(client: TestClient, user1_token: str, user2_token: str):
    created = client.post(
        "/api/v1/wildcards/", json={"name": "secret", "values": ["x"]}, headers=_auth(user1_token)
    ).json()
    wid = created["id"]

    # user2 cannot read or modify user1's private wildcard, and does not see it in their list.
    assert client.get(f"/api/v1/wildcards/i/{wid}", headers=_auth(user2_token)).status_code == 403
    assert (
        client.patch(
            f"/api/v1/wildcards/i/{wid}", json={"values": ["y"]}, headers=_auth(user2_token)
        ).status_code
        == 403
    )
    assert client.delete(f"/api/v1/wildcards/i/{wid}", headers=_auth(user2_token)).status_code == 403
    assert client.get("/api/v1/wildcards/", headers=_auth(user2_token)).json() == []


def test_public_wildcard_is_visible_to_others(client: TestClient, user1_token: str, user2_token: str):
    client.post(
        "/api/v1/wildcards/",
        json={"name": "shared", "values": ["a"], "is_public": True},
        headers=_auth(user1_token),
    )
    names = [w["name"] for w in client.get("/api/v1/wildcards/", headers=_auth(user2_token)).json()]
    assert "shared" in names


def test_import_txt_and_json(client: TestClient, user1_token: str):
    files = [
        ("files", ("animals.txt", "cat\ndog\n\nbird\n", "text/plain")),
        ("files", ("colors.json", '{"colors": ["red", "blue"]}', "application/json")),
    ]
    r = client.post("/api/v1/wildcards/import", files=files, headers=_auth(user1_token))
    assert r.status_code == status.HTTP_200_OK, r.text
    by_name = {w["name"]: w["values"] for w in r.json()}
    assert by_name["animals"] == ["cat", "dog", "bird"]
    assert by_name["colors"] == ["red", "blue"]


def test_import_unsupported_filetype(client: TestClient, user1_token: str):
    files = [("files", ("bad.csv", "a,b,c", "text/csv"))]
    r = client.post("/api/v1/wildcards/import", files=files, headers=_auth(user1_token))
    assert r.status_code == status.HTTP_415_UNSUPPORTED_MEDIA_TYPE


def test_created_wildcard_resolves_in_dynamicprompts(
    client: TestClient, user1_token: str, mock_invoker: Invoker
):
    """A wildcard created by the user must resolve when expanding a prompt that references it."""
    client.post(
        "/api/v1/wildcards/",
        json={"name": "animals", "values": ["cat", "dog", "bird"]},
        headers=_auth(user1_token),
    )
    r = client.post(
        "/api/v1/utilities/dynamicprompts",
        json={"prompt": "a __animals__"},
        headers=_auth(user1_token),
    )
    assert r.status_code == status.HTTP_200_OK, r.text
    body = r.json()
    assert body["error"] is None
    assert sorted(body["prompts"]) == ["a bird", "a cat", "a dog"]
