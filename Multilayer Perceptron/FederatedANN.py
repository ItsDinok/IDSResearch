import flwr as fl
import tensorflow as tf
import numpy as np
import ANNHelper
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings
import matplotlib.pyplot as plt
import math
import sys
warnings.simplefilter('ignore')

# Define model
def CreateModel(features):
    model = Sequential()
    model.add(Dense(features, input_dim=features, kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(int(math.ceil(features/2)), kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(int(math.ceil((features/2)/2)), kernel_initializer='he_normal'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall', 'AUC'])
    model.summary()
    return model


# Flower client class definition
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self, config=None):
        # Return model parameters (weights)
        return self.model.get_weights()

    def set_parameters(self, parameters):
        # Set model parameters (weights)
        self.model.set_weights(self.get_parameters())

    def fit(self, parameters, config):
        # Update local model weights and train the model
        print("Parameters before setting: ", parameters)
        self.set_parameters(parameters)
       
        # Train model
        model.fit(xTrain, yTrain)
        print(f"Training finished for round {config['server_round']}.")
        trainedParameters = self.get_parameters()

        # Set parameters and record results
        self.set_parameters(parameters)
        self.model.fit(xTrain, yTrain, epochs=2, batch_size=4096, verbose=0)
        results = self.model.evaluate(xTrain, yTrain, verbose=0)
        print(f"Local accuracy: {results[1]}")
        print(f"Local precision: {results[2]}")
        print(f"Local recall: {results[3]}")
        return self.get_parameters(), len(xTrain), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        global yTest
        
        # Collect results
        yTest = yTest.astype(int)
        yProb = model.predict(xTest)
        yPred = (yProb > 0.5).astype(int)
        loss = log_loss(yTest, yProb, labels=[0,1])
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(yTest, yPred)
        precision = precision_score(yTest, yPred, average='macro')
        recall = recall_score(yTest, yPred, average='macro')
        f1 = f1_score(yTest, yPred, average='macro')
        AUCROC = roc_auc_score(yTest, yProb)

        performanceMetrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "AUCROC": AUCROC
        }
        
        # Display evaluation metrics
        line = "-" * 21
        print(line)
        print(f"Accuracy: {accuracy:.8f}")
        print(f"Precision: {precision:.8f}")
        print(f"Recall: {recall:.8f}")
        print(f"F1 Score: {f1:.8f}")
        print(f"AUCROC: {AUCROC:.8f}")

        return loss, len(xTest), performanceMetrics

if __name__ == "__main__":
    fileNumber = __file__[-5:].replace(".py", "")
    if fileNumber[0].isdigit():
        clientID = int(fileNumber)
    else:
        clientID = int(fileNumber[1])


    print(f"Client: {clientID}")

    # Get dataset for local model
    # TODO: Consider making this its own helper class
    if len(sys.argv) > 1:
        if sys.argv[1] == "-t":
            xTrain, xTest, yTrain, yTest =  ANNHelper.LoadTransfer(clientID-1)
        else:
            xTrain, xTest, yTrain, yTest = ANNHelper.LoadVertical(clientID-1, int(sys.argv[1]))
    else:    
        xTrain, xTest, yTrain, yTest = ANNHelper.LoadDataset(clientID-1)
    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest = scaler.fit_transform(xTest)

    # Print label distributions
    unique, counts = np.unique(yTrain, return_counts=True)
    trainCounts = dict(zip(unique, counts))
    print("Label distribution in the training set:", trainCounts)

    unique, counts = np.unique(yTest, return_counts=True)
    testCounts = dict(zip(unique, counts))
    print("Label distribution in the test set:", testCounts, "\n")

    model = CreateModel(xTrain.shape[1])
    fl.client.start_numpy_client(server_address='localhost:8080', client=FlowerClient(model))

