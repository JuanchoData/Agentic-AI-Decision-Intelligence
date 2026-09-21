from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_validation():
    response = client.post(
        "/query",
        json={
            "query": ""
        },
    )

    assert response.status_code == 422