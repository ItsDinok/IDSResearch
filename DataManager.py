import pandas as pd
import numpy as np
from typing import List

from sklearn.model_selection import train_test_split

def DataLoader(nodes, clientID):
    df = pd.read_csv()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split dataset into n entries
    # TODO: Make seed dynamic
    np.random.seed(42)
    randomChoice = np.random.choice(X.index, (len(X) % nodes), replace = False)
    X = X.drop(randomChoice)
    y = y.drop(randomChoice)

    # Split dataset into n subsets for n clients
    Xsplit, ySplit = np.split(X, nodes), np.split(y, nodes)
    xSet = []
    ySet = []
    for i in range(nodes):
        xSet.append(Xsplit[i])
        ySet.append(ySplit[i])

    # Set the train test split
    xTrain, yTrain, xTest, yTest = [], [], [], []
    trainSize = 0.8

    xTrain, xTest, yTrain, yTest = train_test_split(xSet(clientID), ySet(clientID), train_size=trainSize, random_state=42)

    return [xTrain, xTest, yTrain, yTest]