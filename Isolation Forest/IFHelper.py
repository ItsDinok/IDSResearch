import pandas as pd
import numpy as np
import math
import Datacleaner
import TransferLearning
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from typing import List
from sklearn.preprocessing import LabelEncoder

def LoadDataset(clientID : int):
    path = "" + f"{clientID}.csv"
    df = pd.read_csv(path)

    benign = df[df.iloc[:, -1] == 1]
    target = df[df.iloc[:, -1] == 0]
    targetCount = int(0.1 * len(benign))
    balancedTarget = target.sample(n=targetCount, replace=True, random_state=42)
    df = pd.concat([benign, balancedTarget]).sample(frac=1, random_state=42)

    # Prepare dataset for export
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = Datacleaner.Normalise(X)
    le = LabelEncoder()
    y = le.fit_transform(y)
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8, random_state=42)
    return xTrain, xTest, yTrain, yTest


def LoadTransfer(clientID):
    path = "" + f"dataset{clientID}.csv"
    df = pd.read_csv(path)
    labelEncoder = LabelEncoder()
    labelEncoder.classes = np.array(['mirai-greeth_flood', 'benign_traffic'])
    df.iloc[:, -1] = labelEncoder.fit_transform(df.iloc[:, -1])
    df = TransferLearning.PrepareData(df)
    print(df.iloc[:, -1])
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    return xTrain, xTest, yTrain, yTest



def LoadVertical(clientID: int, clientCount: int):
    xTrain, xTest, yTrain, yTest = LoadDataset(clientID)
    print("Original shape:", xTrain.shape[1])

    xTrain = pd.DataFrame(xTrain)
    xTest  = pd.DataFrame(xTest)

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

    return xTrain, xTest, yTrain, yTest

def get_params(model: IsolationForest) -> List[np.ndarray]:
    params = [model.n_estimators]
    return params


def set_params(model: IsolationForest, params: List[np.ndarray]) -> IsolationForest:
    return model
    
