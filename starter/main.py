# Put the code for your API here.
from fastapi import FastAPI, HTTPException
from typing import Dict
from pydantic import BaseModel, Field
import pandas as pd

import sys
sys.path.insert(1, './starter/starter/ml')
import data
import model

  
#TODO: Add classes to incorporate categorical fields
#class Profession(str, Enum):
#   DS = "data scientist"
#   MLE = "machine learning scientist"
#   RS = "research scientist"
    
class PredictionPayload(BaseModel):
    age: int = Field(default=39)
    workclass: str = Field(default="Self-emp-not-inc")
    fnlgt: int = Field(default=83311)
    education: str = Field(default="Bachelors")
    education_num: int = Field(default=13, alias="eductaion-num")
    marital_status: str = Field(default="Married-civ-spouse", alias="marital-status")
    occupation: str = Field(default="Handlers-cleaners")
    relationship: str = Field(default="Not-in-family")
    race: str = Field(default="White")
    sex: str = Field(default="Male")
    capital_gain: int = Field(default=0, alias="capital-gain")
    capital_loss: int = Field(default=0, alias="capital-loss") 
    hours_per_week: int = Field(default=40, alias="hours-per-week")
    native_country: str = Field(default="United-States", alias="native-country")
    
    # usage of a validator for data_validation might be good. Atm validation is done withing prediction function 
    #@validator('name')
    #def name_must_contain_space(cls, v):
    #    if ' ' not in v:
    #        raise ValueError('Name must contain a space for first and last name.')
    #    return v

app = FastAPI()
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
encoder,lb = data.read_preprocessors("starter/model") # type: ignore
clf = model.read_model("starter/model") # type: ignore

@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Hello World"}

# Route to invoke a prediction
@app.post("/predict/{age}/{workclass}/{fnlgt}/{education}/{education_num}/{marital_status}/{occupation}/{relationship}/{race}/{sex}/{capital_gain}/{capital_loss}/{hours_per_week}/{native_country}")
def make_prediction(features: PredictionPayload) -> Dict[str, str]:
    # validate data
    if features.age < 0 or features.age > 100:
        raise HTTPException(status_code=400, detail=f"Age mus be between 0 and 100, got {features.age}")
    df = pd.DataFrame([features.dict(by_alias=True)])
       
    processed_features = data.process_data(
        df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb)[0]
    prediction = clf.predict(processed_features)[0]
    response = "<=50K" if prediction == 0 else ">50K" 
    return {"Predicted Income": response}
