"""Auth + per-user portfolio isolation tests.

Runs against a throwaway SQLite DB (no network). Env must be set before the app
is imported, since the engine is built at import time from settings.
"""
import os
import tempfile

os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(tempfile.mkdtemp(), "auth_test.db")
os.environ["SECRET_KEY"] = "test-secret-key-at-least-32-bytes-long-xxxxx"

import pytest
from fastapi.testclient import TestClient

from backend.main import app

ALICE = {"email": "alice@example.com", "password": "supersecret1"}
BOB = {"email": "bob@example.com", "password": "hunter2hunter"}
PF = {"name": "Tech", "market": "NASDAQ",
      "holdings": [{"ticker": "AAPL", "weight": 0.6}, {"ticker": "MSFT", "weight": 0.4}]}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _token(client, creds):
    r = client.post("/auth/login", json=creds)
    return r.json()["access_token"]


def test_register_and_login(client):
    r = client.post("/auth/register", json=ALICE)
    assert r.status_code == 201
    assert r.json()["access_token"]
    assert r.json()["email"] == ALICE["email"]
    # login works and returns a fresh token
    assert client.post("/auth/login", json=ALICE).status_code == 200


def test_duplicate_email_rejected(client):
    assert client.post("/auth/register", json=ALICE).status_code == 409


@pytest.mark.parametrize("payload,code", [
    ({"email": "weak@example.com", "password": "short"}, 422),      # < 8 chars
    ({"email": "not-an-email", "password": "longenough1"}, 422),    # bad email
])
def test_signup_validation(client, payload, code):
    assert client.post("/auth/register", json=payload).status_code == code


def test_login_failures_are_generic(client):
    wrong = client.post("/auth/login", json={**ALICE, "password": "wrongpass1"})
    ghost = client.post("/auth/login", json={"email": "ghost@example.com", "password": "whatever1"})
    assert wrong.status_code == ghost.status_code == 401
    assert wrong.json()["detail"] == ghost.json()["detail"] == "Incorrect email or password."


def test_protected_routes_need_a_token(client):
    assert client.post("/portfolios", json=PF).status_code in (401, 403)
    assert client.get("/portfolios").status_code in (401, 403)
    assert client.get("/auth/me", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_users_are_isolated(client):
    client.post("/auth/register", json=BOB)
    ha = {"Authorization": f"Bearer {_token(client, ALICE)}"}
    hb = {"Authorization": f"Bearer {_token(client, BOB)}"}

    pid = client.post("/portfolios", json=PF, headers=ha).json()["id"]

    assert len(client.get("/portfolios", headers=ha).json()) == 1
    assert client.get("/portfolios", headers=hb).json() == []
    assert client.get(f"/portfolios/{pid}", headers=hb).status_code == 404
    assert client.get(f"/portfolios/{pid}", headers=ha).status_code == 200
    # ownership is checked before any price fetch, so this never hits the network
    assert client.post("/rebalance", json={"portfolio_id": pid, "method": "max_sharpe"},
                       headers=hb).status_code == 404


def test_delete_is_user_scoped(client):
    ha = {"Authorization": f"Bearer {_token(client, ALICE)}"}
    hb = {"Authorization": f"Bearer {_token(client, BOB)}"}
    pid = client.post("/portfolios", json={**PF, "name": "ToDelete"}, headers=ha).json()["id"]

    assert client.delete(f"/portfolios/{pid}").status_code in (401, 403)   # no token
    assert client.delete(f"/portfolios/{pid}", headers=hb).status_code == 404  # not bob's
    assert client.get(f"/portfolios/{pid}", headers=ha).status_code == 200     # still there
    assert client.delete(f"/portfolios/{pid}", headers=ha).status_code == 204  # alice deletes
    assert client.get(f"/portfolios/{pid}", headers=ha).status_code == 404     # gone
    assert client.delete(f"/portfolios/{pid}", headers=ha).status_code == 404  # idempotent-ish


def test_rename_is_user_scoped(client):
    ha = {"Authorization": f"Bearer {_token(client, ALICE)}"}
    hb = {"Authorization": f"Bearer {_token(client, BOB)}"}
    pid = client.post("/portfolios", json={**PF, "name": "OldName"}, headers=ha).json()["id"]

    assert client.patch(f"/portfolios/{pid}", json={"name": "X"}).status_code in (401, 403)  # no token
    assert client.patch(f"/portfolios/{pid}", json={"name": "Hijack"}, headers=hb).status_code == 404  # not bob's
    assert client.patch(f"/portfolios/{pid}", json={"name": ""}, headers=ha).status_code == 422  # empty rejected

    r = client.patch(f"/portfolios/{pid}", json={"name": "NewName"}, headers=ha)
    assert r.status_code == 200 and r.json()["name"] == "NewName"
    assert client.get(f"/portfolios/{pid}", headers=ha).json()["name"] == "NewName"  # persisted


# A full optimize result as the frontend would snapshot it at save time.
SNAPSHOT = {
    "method": "max_sharpe",
    "allocations": [{"ticker": "AAPL", "weight": 0.6}, {"ticker": "MSFT", "weight": 0.4}],
    "metrics": {"expected_return": 0.21, "volatility": 0.18, "sharpe": 0.95, "risk_free_rate": 0.043},
    "frontier": [{"volatility": 0.15, "expected_return": 0.1, "sharpe": 0.4},
                 {"volatility": 0.2, "expected_return": 0.22, "sharpe": 0.9}],
}


def test_snapshot_round_trips(client):
    """A saved snapshot comes back byte-for-byte on GET and in the list, so the
    UI can restore the exact charts/returns/risk later."""
    ha = {"Authorization": f"Bearer {_token(client, ALICE)}"}
    pid = client.post("/portfolios", json={**PF, "name": "Snap", "snapshot": SNAPSHOT},
                      headers=ha).json()["id"]

    got = client.get(f"/portfolios/{pid}", headers=ha).json()
    assert got["snapshot"] == SNAPSHOT

    listed = [p for p in client.get("/portfolios", headers=ha).json() if p["id"] == pid][0]
    assert listed["snapshot"] == SNAPSHOT


def test_snapshot_is_optional(client):
    """Saving without a snapshot (legacy/omitted) still works and reports None."""
    ha = {"Authorization": f"Bearer {_token(client, ALICE)}"}
    pid = client.post("/portfolios", json={**PF, "name": "NoSnap"}, headers=ha).json()["id"]
    assert client.get(f"/portfolios/{pid}", headers=ha).json()["snapshot"] is None
