import pandas as pd
import numpy as np
import Datacleaner
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from typing import List


def LoadDataset(clientID : int):
    """
    This is something I call a set-state data loader. A separate splitter file will
    shuffle and split the data into n csv files. This can be found in the 'Datacleaner.py' file
    
    This has been done to avoid extreme memory usage that is found in other implementations
    """
    # NOTE: Path needs to be inserted
    path = "" + f"{clientID}.csv"
    df = pd.read_csv(path)

    # Prepare dataset for export
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X = Datacleaner.Normalise(X)
    y = DataCleaner.LabelEncode(y)
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8, random_state=42)
    return xTrain, yTrain, xTest, yTest


def get_params(model: IsolationForest) -> List[np.ndarray]:
    params = model[
        model.verbose,
        model.n_estimators
    ]
    return params


def set_params(model: IsolationForest, params: List[np.ndarray]) -> IsolationForest:
    model.verbose = params[0]
    model.n_estimators = params[1]
    return model
    
