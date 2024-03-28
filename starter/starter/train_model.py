# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Add the necessary imports for the starter code.
import sys
sys.path.insert(1, './ml')
from data import process_data, write_preprocessors
from model import train_model, write_model
# Add code to load in the data.
data = pd.read_csv('../data/census_clean.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

write_preprocessors(encoder, lb, '../model')

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)
# Train and save a model.
clf = train_model(X_train, y_train)

write_model(clf, '../model')


