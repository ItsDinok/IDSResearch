import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from scikeras.wrappers import KerasClassifier
import warnings
import math
import sys
warnings.simplefilter('ignore')

def CreateModel(input_dim):
    features = input_dim

    model = Sequential()
    model.add(Dense(features, input_dim=features, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    # This dynamically calculates the number of nodes in the next layer
    nextLayer = int(math.ceil(features/2))
    model.add(Dense(nextLayer, kernel_initializer = 'he_normal'))
    model.add(LeakyReLU(alpha=0.01))

    nextLayer = int(math.ceil(nextLayer/2))
    model.add(Dense(nextLayer, kernel_initializer = 'he_normal'))
    model.add(LeakyReLU(alpha=0.01))

    model.add(Dense(1, activation='sigmoid'))
    model.compile (
        optimizer = 'sgd',
        loss = 'binary_crossentropy',
        metrics = ['accuracy', 'Precision', 'Recall', 'AUC']
    )
    model.summary()
    return model


def Main():
    df = pd.read_csv("C:/Users/markd/Desktop/FL Tools/cleaned.csv")
    df = df.drop(columns=['Tot sum'])
    ss = StandardScaler()
    X = df.iloc[:, :-1]
    print(X.columns)
    X = ss.fit_transform(X)

    y = df.iloc[:, -1]
    le = LabelEncoder()
    y = le.fit_transform(y)

    model = CreateModel(X.shape[1])
    kerasModel = KerasClassifier(model=CreateModel, input_dim=X.shape[1], epochs=5, batch_size=512, verbose=1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(kerasModel, X, y, cv=cv, scoring="accuracy")
    print("Cross validation scores:", scores)
    print("Mean accuracy:", scores.mean())

    testSet = pd.read_csv("C:/Users/markd/Desktop/FL Tools/Testing Set.csv")
    xTest = testSet.iloc[:, :-1]
    yTest = testSet.iloc[:, -1]

    results = model.evaluate(xTest, yTest, verbose=1)
    print("Test loss:", results[0])
    print("Test accuracy:", results[1])
    print("Test precision:", results[2])
    print("Test Recall:", results[3])
    print("Test AUC:", results[4])

if __name__ == "__main__":
    Main()
