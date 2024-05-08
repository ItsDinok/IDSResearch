# Client code

from logging import makeLogRecord
from typing import Dict
from flwr.common import NDArrays, Scalar
import numpy as np
import flwr as fl
from scipy.special import y1p_zeros
from scipy.stats import entropy

from sklearn.svm import SVC
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
from sklearn import datasets
from sklearn.model_selection import train_test_split
from typing import List

# TODO: Find out what this does
import warnings
warnings.simplefilter('ignore')


def GetParameters(model: SVC):
    parameters = [
        model.degree,
        model.gamma,
        model.coef0
        ]
    
    return parameters


def SetParameters(model: SVC, parameters: List[np.ndarray]):
    model.degree = int(parameters[0])
    model.gamma = parameters[1]
    model.coef0 = int(parameters[2])
    
    return model
    


# Create flwr client
class FlowerClient(fl.client.NumPyClient):
    # Get current local model parameters
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        print(f"Client {client_id} recieved the parameters")
        return GetParameters(model)
    
    
    def evaluate(self, parameters, config):
        SetParameters(model, parameters)
        
        yPred = model.predict(xTest)
        loss = log_loss(yTest, yPred, labels=[0, 1])
        
        accuracy = accuracy_score(yTest, yPred)
        precision = precision_score(yTest, yPred, average='weighted')
        recall = recall_score(yTest, yPred, average='weighted')
        f1 = f1_score(yTest, yPred, average='weighted')
        
        line = "-" * 21
        print(line)
        
        return loss, len(xTest), {"Accuracy" : accuracy, "Precision" : precision, "Recall" : recall, "F1_Score" : f1}
    

# Entry point
if __name__ == "__main__":
    client_id = 1
    print(f"Client {client_id}:\n")
    
    # Get dataset for local model
    # NOTE: This needs to be changed when federated
    X, y = datasets.load_iris().data, datasets.load_iris().target
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, random_state=42,train_size=0.8)
    
    # Print Label Distribution
    unique,  counts = np.unique(yTrain, return_counts=True)
    trainCounts = dict(zip(unique, counts))
    print("Label Distribution in the training set:", trainCounts)
    
    unique, counts = np.unique(yTest, return_counts = True)
    testCounts = dict(zip(unique, counts))
    print("Label distribution in the training set:", testCounts)
    
    # Create and fit the local model
    model = SVC( 
        kernel='rbf',
        tol=0.001,
        class_weight=None,
        verbose=False,
    )
    
    model.fit(xTrain, yTrain)
    
    # Start Client
    fl.client.start_numpy_client(server_address="LOCALHOST:8080", client=FlowerClient())