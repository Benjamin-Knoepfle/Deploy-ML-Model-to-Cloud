import json
import requests

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

URL = "https://deploy-ml-model-to-cloud.onrender.com"
r = requests.post(URL+"/predict/", data=json.dumps(features))

print(r.status_code)
print(json.loads(r.text))
