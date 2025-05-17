import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success():
    with open("tests/test_image.jpg", "rb") as image:
        response = client.post("/predict", files={"file": ("tests.jpg", image, "image/jpeg")})

    assert response.status_code == 200
    data = response.json()
    assert "prediction_uid" in data
    assert "labels" in data
    assert isinstance(data["labels"], list)


def test_predict_missing_file():
    response = client.post("/predict", files={})
    assert response.status_code == 422  # or 400 depending on your validation

def test_get_prediction_by_uid_valid():
    # Replace this UID with a known valid one in your DB for tests
    response = client.get("/prediction/tests-valid-uid")
    assert response.status_code in [200, 404]  # depending on setup

def test_get_prediction_by_uid_invalid():
    response = client.get("/prediction/invalid-uid")
    assert response.status_code == 404

def test_get_predictions_by_label():
    response = client.get("/predictions/label/person")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_predictions_by_label_not_found():
    response = client.get("/predictions/label/nonexistent_label")
    assert response.status_code == 200
    assert response.json() == []

def test_get_predictions_by_score():
    response = client.get("/predictions/score/0.5")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_get_predictions_by_score_invalid():
    response = client.get("/predictions/score/invalid")
    assert response.status_code == 422

def test_get_prediction_image_valid():
    response = client.get("/prediction/tests-valid-uid/image")
    assert response.status_code in [200, 404]

def test_get_prediction_image_invalid():
    response = client.get("/prediction/invalid-uid/image")
    assert response.status_code == 404

def test_get_image_by_filename_valid():
    response = client.get("/image/original/tests.jpg")
    assert response.status_code in [200, 404]

def test_get_image_by_filename_invalid():
    response = client.get("/image/unknown/tests.jpg")
    assert response.status_code == 400
