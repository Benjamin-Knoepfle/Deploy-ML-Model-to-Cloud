# Script to train machine learning model.

# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import sys
sys.path.insert(1, './ml')
import data  # noqa
import model # noqa

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

data_ = pd.read_csv('../data/census.csv')
data_.columns = data_.columns.str.strip()
data_[cat_features] = data_[cat_features].apply(lambda x: x.str.strip())
data_.to_csv('../data/census_clean.csv', index=False)

# Add code to load in the data.
data_ = pd.read_csv('../data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data_, test_size=0.20)

X_train, y_train, encoder, lb = data.process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

data.write_preprocessors(encoder, lb, '../model')

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = data.process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder, lb=lb
)
# Train and save a model.
clf = model.train_model(X_train, y_train)
model.write_model(clf, '../model')

# Test the model performance
predictions = model.inference(clf, X_test)
precision, recall, fbeta = model.compute_model_metrics(
    y_test, predictions)  # type: ignore
print(f"Model score on test data: Precision = {precision}")
print(f"Model score on test data: Recall = {recall}")
print(f"Model score on test data: fbeta = {fbeta}")
performances_on_slices = model.performance_on_dataslices(
    test, y_test, predictions)  # type: ignore

with open('slice_output.txt', 'w') as fp:
    json.dump(performances_on_slices, fp)
