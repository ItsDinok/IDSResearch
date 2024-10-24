import pandas as pd
import numpy as np
import Datacleaner
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List

def LoadDataset(clientID : int):
    """
    This is something I call a set-state data loader. A separate splitter file will
    shuffle and split the data into n csv files. This can be found in the 'Datacleaner.py' file
    
    This has been done to avoid extreme memory usage that is found in other implementations
    """

    path = "" + f"{clientID}.csv"
    df = pd.read_csv(path)

    # Prepare the data for export
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8, random_state=42)
    return xTrain, yTrain, xTest, yTest


def LoadVerticalDataset(clientID: int, totalClients: int):
    df = pd.read_csv(path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Perform vertical partitioning based on client ID
    featureNumber = X.shape[1]
    featuresPerClient = featureNumber / totalClients

    # Determine which columns the node gets
    startCol = clientID * featuresPerClient
    endCol = clientID + featuresPerClient if clientID < totalClients -1 else featureNumber

    xClient = X.iloc[:, startCol:endCol]

    y = Datacleaner.LabelEncode(y)

    # Split data into train and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, training_size=0.8, random_state=42)
    return xTrain, xTest, yTrain, yTest

def get_params(model: RandomForestClassifier) -> List[np.ndarray]:
    parameters = [
        model.n_estimators
        model.max_depth,
        model.min_samples_split,
        model.min_samples_leaf
    ]
    return parameters


def set_params(model: RandomForestClassifier, parameters: List[np.ndarray]) -> RandomForestClassifier:
    model.n_estimators = int(parameters[0])
    model.max_depth = int(parameters[1])
    model.min_samples_split = int(parameters[2])
    model.min_samples_leaf = int(parameters[3])
    return model

    
