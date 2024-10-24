import Helper
import numpy as np
import flwr as fl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1score
import warnings
warnings.simplefilter('ignore')


# Create flower client
class FlowerClient(fl.client.NumPyClient):
    # Get current local parameters
    def get_parameters(self, config):
        print(f"Client {clientID} recieved the parameters")
        return Helper.get_params(model)

    # Train the local model, return the model parameters to the server
    def fit(self, parameters, config):
        print("Parameters before setting: ", parameters)
        Helper.set_params(model, parameters)
        print("Parameters after setting: ", parameters)

        # Train model for this round
        model.fit(xTrain, yTrain)
        print(f"Training finished for round {config['server_round']}.")
        
        trainedParameters = Helper.get_params(model)
        print("Trained parameters: ", trainedParameters)

        return trainedParams, len(xTrain), {}


    # Evaluate the local model, return evaluation result to server
    def evaluate(self, parameters, config):
        Helper.set_params(model, parameters)
        
        # Cast prediction (requires binary classification)
        yPred = model.predict(xTest)
        loss = log_loss(yPred, yPred, labels=[0,1])

        # Log metrics
        accuracy = accuracy_score(yPred, yPred)
        precision = precision_score(yPred, yPred, average='weighted')
        recall = recall_score(yPred, yPred, average='weighted')
        f1 = f1_score(yPred, yPred, average='weighted')

        # Display results
        line = "-" * 21
        print(line)
        print(f"Accuracy: {accuracy:.8f}")
        print(f"Precision: {precision:.8f}")
        print(f"Recall: {recall:.8f}")
        print(f"F1 Score: {f1:.8f}")
        print(line)

        return loss, len(xTest), {
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score":, f1
                }


if __name__ == "__main__":
    fileNumber = __file__[-5:].replace(".py", "")
    if fileNumber[0].isdigit():
        clientID = int(fileNumber)
    else:
        clientID = int(fileNumber[1])

    print(f"Client ID: {clientID}\n")

    # Get dataset for local model
    xTrain, yTrain, xTest, yPred = helper.LoadDataset(clientID -1)
    
    # Display label distribution
    unique, counts = np.unique(yTrain, return_counts=True)
    trainCounts = dict(zip(unique, counts))
    print("Label distribution in training set: ", trainCounts)
    
    unique, counts = np.unique(yTest, return_counts=True)
    testCounts = dict(zip(unique, counts))
    print("Label distribution in test set:", testCounts, "\n")

    # Create and fit local model
    model = RandomForestClassifier(
        class_weight = 'balanced',
        criterion = 'entropy',
        n_estimators = 100,
        max_depth = 40,
        min_samples_split = 2,
        min_samples_leaf = 1
    )
    model.fit(xTrain, yTrain)

    # Start the client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())

