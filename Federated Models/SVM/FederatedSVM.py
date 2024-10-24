import SVMHelper
import numpy as np
import flwr as fl
from sklearn.svm import LinearSVC
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.simplefilter('ignore')


# Create flower client
class FlowerClient(fl.client.NumPyClient):
    # Get current model parameters
    def get_parameters(self, config):
        print(f"Client {clientID} recieved the parameters.")
        return SVMHelper.get_params(model, parameters)

    # Train the local model, return parameters to the server
    def fit(self, parameters, config):
        print("Parameters before setting:", parameters)
        SVMHelper.set_params(model, parameters)
        print("Parameters after setting:", model.get_params())

        model.fit(xTrain, yTrain)
        print(f"Training finished for round {config['server_round']}.")
        
        trainedParameters = SVMHelper.get_params(model)
        print(f"Trained parameters: ", trainedParameters)
        return trainedParameters, len(xTrain), {}

    # Evaluate the local model and return result to server
    def evaluate(self, parameters, config):
        SVMHelper.set_params(model, parameters)

        yPred = model.predict(xTest)
        loss = log_loss(yTest, yPred, labels=[0,1])

        accuracy = accuracy_score(yTest, yPred)
        precision = precision_score(yTest, yPred)
        recall = recall_score(yTest, yPred)
        f1 = f1_score(yTest, yPred)

        line = "-" * 21
        print(line)
        print(f"Accuracy: {accuracy.8f}")
        print(f"Precision: {precision.8f}")
        print(f"Recall: {recall.8f}")
        print("F1 Score: {f1.8f}")
        print(line)

        return loss, len(xTest), {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1
                }


if __name__ == "__main__":
    fileNumber = __file__[-5:].replace(".py", "")
    if fileNumber[0].isdigit():
        clientID = int(fileNumber)
    else:
        clientID = int(fileNumber[1])

    print(f"Client: {clientID}")

    # Get dataset for local model
    xTrain, xTest, yTrain, yTest = SVMHelper.loadDataset(clientID - 1)

    # Print label distributions
    unique, counts = np.unique(yTrain, return_counts=True)
    trainCounts = dict(zip(unique, counts))
    print("Label distribution in the training set:", trainCounts)

    unique, counts = np.unique(yTest, return_counts=True)
    testCounts = dict(zip(unique, counts))
    print("Label distribution in the test set:", testCounts, "\n")

    # Create and fit local model
    model = LinearSVC(
        penalty = 'l2',
        loss = 'hinge',
        max_iter = 50,
        C = 1
    )
    model.fit(xTrain, yTrain)

    # Start the client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())

