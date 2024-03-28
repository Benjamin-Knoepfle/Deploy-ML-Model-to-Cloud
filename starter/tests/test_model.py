import sys
sys.path.insert(1, 'starter/starter/ml')
from model import train_model, write_model
import data
import pandas as pd

def test_model_creation():
    train = pd.read_csv('starter/tests/test_data/census_test.csv')
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
    X_train, y_train, encoder, lb = data.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X_train, y_train)
    assert model != None

#def test_write_model():
#    model = None
#    dest_pth = 'starter/tests/test_data'
#    write_model(model, dest_pth)
   