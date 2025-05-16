import numpy as np
import flwr as fl
import Datacleaner
import IFHelper
import sys
from sklearn.ensemble import IsolationForest
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
import math
import TransferLearning
warnings.simplefilter('ignore')


# Create flower client
class FlowerClient(fl.client.NumPyClient):
    # Get local model parameters
    def get_parameters(self, config):
        print(f"Client {clientID} recieved the parameters.")
        return IFHelper.get_params(model)


    # Train the local model, return parameters to the server
    def fit(self, parameters, config):
        print("Parameters before setting:", parameters)
        IFHelper.set_params(model, parameters)
        print("Parameters after setting:", parameters)

        model.fit(xTrain, yTrain)
        print("Training finished for round {config['server_round']}.")

        trainedParameters = IFHelper.get_params(model)
        print("Trained parameters: ", trainedParameters)

        return trainedParameters, len(xTrain), {}


    # Evaluate the local model, 
    def evaluate(self, parameters, config):
        IFHelper.set_params(model, parameters)

        yPred = model.predict(xTest)
        yProb = yPred
        yPred = -yPred
        yPred = (yPred < 0).astype(int)
        loss = log_loss(yTest, yPred, labels=[0,1])

        accuracy = accuracy_score(yTest, yPred)
        precision = precision_score(yTest, yPred)
        recall = recall_score(yTest, yPred)
        f1 = f1_score(yTest, yPred)
        aucroc = roc_auc_score(yTest, yProb)

        performanceMetrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "AUCROC": aucroc
        }


        line = "-" * 21
        print(line)
        print(f"Accuracy: {accuracy:.8f}")
        print(f"Precision: {precision:.8f}")
        print(f"Recall: {recall:.8f}")
        print(f"F1 Score: {f1:.8f}")
        print(f"AUC ROC: {aucroc:.8f}")

        return loss, len(xTest), performanceMetrics


if __name__ == "__main__":
    fileNumber = __file__[-5:].replace(".py", "")
    if fileNumber[0].isdigit():
        clientID = int(fileNumber)
    else:
        clientID = int(fileNumber[1])

    if len(sys.argv) > 1:
        if sys.argv[1] == "-t":
            xTrain, xTest, yTrain, yTest = IFHelper.LoadTransfer(clientID-1)
        else:
            xTrain, xTest, yTrain, yTest = IFHelper.LoadVertical(clientID-1, int(sys.argv[1]))
    else:
        xTrain, xTest, yTrain, yTest = IFHelper.LoadDataset(clientID-1)

    scaler = StandardScaler()
    xTrain = scaler.fit_transform(xTrain)
    xTest  = scaler.fit_transform(xTest)

    print(xTest.shape) 
    # Print label distributions
    unique, counts = np.unique(yTrain, return_counts=True)
    trainCounts = dict(zip(unique, counts))
    print("Label distribution in training set:", trainCounts)

    unique, counts = np.unique(yTest, return_counts=True)
    testCounts = dict(zip(unique, counts))
    print("Label distribution in testing set:", testCounts)

    # Create and fit local model
    model = IsolationForest (
        random_state = 42,
        verbose = 0,
        n_estimators = 50,
        contamination = 0.1,
        max_samples=64
    )

    # Start the client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
