import sys
sys.path.insert(1, 'starter/starter/ml')
from sklearn.tree import DecisionTreeClassifier
from model import train_model, write_model, read_model
import data
import pandas as pd

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

def test_model_creation():
    train = pd.read_csv('starter/tests/test_data/census_test.csv')
    X_train, y_train, encoder, lb = data.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    clf = train_model(X_train, y_train)
    assert isinstance(clf, DecisionTreeClassifier)

def test_read_model():
    src_pth = 'starter/tests/test_data'
    clf = read_model(src_pth)
    assert isinstance(clf, DecisionTreeClassifier)
    
def test_inference():
    src_pth = 'starter/tests/test_data'
    features = pd.read_csv(f'{src_pth}/census_test.csv')
    features.drop(['salary'], inplace=True, axis=1)
    encoder, lb = data.read_preprocessors(src_pth)
    clf = read_model(src_pth) 
    
    processed_features = data.process_data(
        features,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb)[0]
    
    prediction = clf.predict(processed_features)

    assert len(features)==len(prediction)
    assert max(prediction)==1
    assert min(prediction)==0 
      
