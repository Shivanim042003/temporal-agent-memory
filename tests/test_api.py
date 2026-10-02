from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "database": "ok",
    }


def test_memory_query_returns_response():
    response = client.post(
        "/memory/query",
        json={
            "query": "What am I using now?",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert "answer" in body
    assert "memories" in body
    assert "query_time" in body

    assert body["memories"] == []


def test_memory_query_accepts_source_id():
    response = client.post(
        "/memory/query",
        json={
            "query": "What am I using now?",
            "source_id": "test-source",
        },
    )

    assert response.status_code == 200


def test_memory_query_rejects_empty_query():
    response = client.post(
        "/memory/query",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422