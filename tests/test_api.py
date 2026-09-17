from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_validation():
    payload = {
        "age": 55,
        "bmi": 29,
        "systolic_bp": 135,
        "hba1c": 6.2,
        "ldl": 125,
        "length_of_stay": 4,
        "prior_admissions": 1,
        "comorbidity_count": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in (200, 503)
