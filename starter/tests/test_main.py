from main import app
from requests import Response
from fastapi.testclient import TestClient
import json
import sys
sys.path.insert(1, 'starter')
sys.path.insert(1, 'starter/starter/ml')

client = TestClient(app)


def test_get_on_root() -> None:
    r: Response = client.get("/")
    assert r.status_code == 200
    assert json.loads(r.text) == {"message": "Hello World"}  # type: ignore


def test_post_prediction_leq_50K():
    features = {
        "age": 36,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 83311,
        "education": "Bachelors",
        "eductaion-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Handlers-cleaners",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    r = client.post("/predict/", data=json.dumps(features))
    assert r.status_code == 200
    assert json.loads(r.text) == {"Predicted Income": "<=50K"}  # type: ignore


def test_post_prediction_ge_50K():
    features = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 116632,
        "education": "Doctorate",
        "eductaion-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    }
    r = client.post("/predict/", data=json.dumps(features))
    assert r.status_code == 200
    assert json.loads(r.text) == {"Predicted Income": ">50K"}  # type: ignore
