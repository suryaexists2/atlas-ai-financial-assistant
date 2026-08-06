"""Health endpoint tests (M0)."""

import pytest


@pytest.mark.asyncio
async def test_health_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["app"] == "Atlas"
    assert body["version"]


@pytest.mark.asyncio
async def test_health_db_reachable(client):
    response = await client.get("/health/db")
    assert response.status_code == 200
    assert response.json()["database"] == "reachable"
