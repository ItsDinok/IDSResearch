import pandas as pd
import time
import Datacleaner
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris


def ConstructModel():
    # Define model 
    model = Sequential()
    # TODO: Change this for the actual number of features
    # Structure
    model.add(Dense(22, input_shape=(4,), kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(13, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def GetDataset():
    # TODO: Change this to be suitable for external data
    iris = load_iris()
    X = iris.data
    y = iris.target
    X = Datacleaner.Normalise(X)
    yBinary = (y != 0).astype(int)

    xTrain, xTest, yTrain, yTest = train_test_split(X, yBinary, test_size=0.2, random_state=42)
    return [xTrain, xTest, yTrain, yTest]


if __name__ == "__main__":
    model = ConstructModel()
    xTrain, xTest, yTrain, yTest = GetDataset()

    model.fit(xTrain, yTrain, epochs=50, batch_size=8, validation_data=(xTest, yTest))
    testLoss, testAcc = model.evaluate(xTest, yTest, verbose=2)
    print(f"Test Accuracy: {testAcc}")

    input()


