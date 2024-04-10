import os
import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

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

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model
    using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def performance_on_dataslices(data, y, preds):
    performances = {}
    for feature in cat_features:
        performances[feature] = performance_on_dataslice(
            data, y, preds, feature)
    return performances


def performance_on_dataslice(data, y, preds, feature):
    values = np.unique(data[feature])
    performances = {}

    data_ = data.copy()
    data_['target'] = y
    data_['predictions'] = preds
    for val in values:
        slice = data_[data[feature] == val]
        precision, recall, fbeta = compute_model_metrics(
            slice['target'], slice['predictions'])
        performances[val] = {
            'precision': precision,
            'recall': recall,
            'fbeta': fbeta
        }
    return performances


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    prediction = model.predict(X)
    return prediction


def write_model(model, dest_pth):
    with open(os.path.join(dest_pth, 'model.pkl'), 'wb') as fp:
        pickle.dump(model, fp)


def read_model(src_pth):
    with open(os.path.join(src_pth, 'model.pkl'), 'rb') as fp:
        model = pickle.load(fp)
    return model
