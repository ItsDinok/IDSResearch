from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from typing import List
import pandas as pd
import numpy as np
import Datacleaner
import TransferLearning
import math

def LoadDataset(clientID : int):
    path = "C:/Users/markd/Desktop/Datasets/" + f"{clientID}.csv"
    df = pd.read_csv(path)

    # Partition dataset 
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Process dataset for ML
    X = Datacleaner.Normalise(X)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Return
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8, random_state=42)
    return xTrain, xTest, yTrain, yTest


# This applies sample and feature space reduction 
def LoadTransfer(clientID):
    path = "C:/Users/markd/Desktop/Datasets" + f"dataset{clientID}.csv"
    df = pd.read_csv(path)
    
    # Encode data for correlation analysis
    labelEncoder = LabelEncoder()
    df.iloc[:, -1] = labelEncoder.fit_transform(df.iloc[:, -1])
    
    # Reduce dataset
    df = TransferLearning.PrepareData(df)
    print(df.iloc[:, -1])
    
    # Split and export
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    return xTrain, xTest, yTrain, yTest


def LoadVertical(clientID: int, clientCount: int):
    # The model needs to be turned back into a pd.df
    xTrain, xTest, yTrain, yTest = LoadDataset(clientID)
    print("Original shape:", xTrain.shape[1])
    xTrain = pd.DataFrame(xTrain)
    xTest = pd.DataFrame(xTest)

    # These need to be more than one in order to get full coverage
    partitionSize = math.ceil(xTrain.shape[1] / clientCount)
    partitionSize = max(partitionSize, 6)
    # The -1 here sets the start to 0
    min_features = 6
    max_features = xTrain.shape[1]
    partitionSize = max(partitionSize, min_features)

    # Determine start and end
    start = partitionSize * clientID
    end = start + partitionSize

    # Wrap around if needed
    if start >= max_features:
        start = (clientID * min_features) % max_features
        end = start + min_features

    # Clamp to the end if it would overflow
    if end > max_features:
        end = max_features
        start = max(0, end - partitionSize)

    # Slice
    xTrain = xTrain.iloc[:, start:end]
    xTest = xTest.iloc[:, start:end]

    print(xTrain.shape)
    return xTrain, xTest, yTrain, yTest

def get_params(model: Sequential) -> List[np.ndarray]:
    params = model.get_weights()
    return params


def set_params(model: Sequential, params: List[np.ndarray]) -> Sequential:
    return model
    
