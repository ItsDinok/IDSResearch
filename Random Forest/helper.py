import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import List


def load_dataset(client_id: int):
    df = pd.read_csv('C:/Users/markd/Desktop/Research/ReducedDDoS.csv')

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split the dataset evenly into thirds, removing the remainders
    np.random.seed(42)
    random_choose = np.random.choice(X.index, (len(X) % 10), replace=False)
    X = X.drop(random_choose)
    y = y.drop(random_choose)

    # Label encode y
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # Split the dataset into 3 subsets for 3 clients
    X_split, y_split = np.split(X, 10), np.split(y, 10)
    X1, y1 = X_split[0], y_split[0]
    X2, y2 = X_split[1], y_split[1]
    X3, y3 = X_split[2], y_split[2]
    X4, y4 = X_split[3], y_split[3]
    X5, y5 = X_split[4], y_split[4]
    X6, y6 = X_split[5], y_split[5]
    X7, y7 = X_split[6], y_split[6]
    X8, y8 = X_split[7], y_split[7]
    X9, y9 = X_split[8], y_split[8]
    X10, y10 = X_split[9], y_split[9]

    del X
    del y

    # Split the training set and testing set in 80% ratio
    X_train, y_train, X_test, y_test = [], [], [], []
    train_size = 0.8

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=train_size, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=train_size, random_state=42)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, train_size=train_size, random_state=42)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, train_size=train_size, random_state=42)
    X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, train_size=train_size, random_state=42)
    X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, train_size=train_size, random_state=42)
    X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, train_size=train_size, random_state=42)
    X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, train_size=train_size, random_state=42)
    X9_train, X9_test, y9_train, y9_test = train_test_split(X9, y9, train_size=train_size, random_state=42)
    X10_train, X10_test, y10_train, y10_test = train_test_split(X10, y10, train_size=train_size, random_state=42)

    X_train.append(X1_train)
    X_train.append(X2_train)
    X_train.append(X3_train)
    X_train.append(X4_train)
    X_train.append(X5_train)
    X_train.append(X6_train)
    X_train.append(X7_train)
    X_train.append(X8_train)
    X_train.append(X9_train)
    X_train.append(X10_train)

    y_train.append(y1_train)
    y_train.append(y2_train)
    y_train.append(y3_train)
    y_train.append(y4_train)
    y_train.append(y5_train)
    y_train.append(y6_train)
    y_train.append(y7_train)
    y_train.append(y8_train)
    y_train.append(y9_train)
    y_train.append(y10_train)

    X_test.append(X1_test)
    X_test.append(X2_test)
    X_test.append(X3_test)
    X_test.append(X4_test)
    X_test.append(X5_test)
    X_test.append(X6_test)
    X_test.append(X7_test)
    X_test.append(X8_test)
    X_test.append(X9_test)
    X_test.append(X10_test)

    y_test.append(y1_test)
    y_test.append(y2_test)
    y_test.append(y3_test)
    y_test.append(y4_test)
    y_test.append(y5_test)
    y_test.append(y6_test)
    y_test.append(y7_test)
    y_test.append(y8_test)
    y_test.append(y9_test)
    y_test.append(y10_test)

    # Each of the following is divided equally into thirds
    return X_train[client_id], y_train[client_id], X_test[client_id], y_test[client_id]


# Look at the RandomForestClassifier documentation of sklearn and select the parameters
# Get the parameters from the RandomForestClassifier
def get_params(model: RandomForestClassifier) -> List[np.ndarray]:
    params = [
        model.n_estimators,
        model.max_depth,
        model.min_samples_split,
        model.min_samples_leaf,
    ]
    return params


# Set the parameters in the RandomForestClassifier
def set_params(model: RandomForestClassifier, params: List[np.ndarray]) -> RandomForestClassifier:
    model.n_estimators = int(params[0])
    model.max_depth = int(params[1])
    model.min_samples_split = int(params[2])
    model.min_samples_leaf = int(params[3])
    return model
