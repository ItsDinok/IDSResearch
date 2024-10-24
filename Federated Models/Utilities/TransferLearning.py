import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import load


def LoadDatasetTransferLearning(clientID: int, modelPath: str):
    # Load dataset for client
    path = "" + f"{clientID}.csv"
    df = pd.read_csv(path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=0.8, random_state=42)

    model = RandomForestClassifier(
        class_weight = 'balanced',
        criterion = 'entropy',
        n_estimators = 100,
        max_depth = 40,
        min_samples_split = 2,
        min_samples_leaf = 1
    )

    model.fit(xTrain, yTrain)
    joblib.dump(model, "trainedModel")

    return model, xTrain, yTrain, xTest, yTest

