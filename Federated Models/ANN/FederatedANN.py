import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Define model
def CreateModel():
    model = Sequential()
    model.add(Dense(22, input_shape=(4,), kernel_intializer='he_normal', activation='relu'))
    model.add(Dense(13, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(7, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='he_normal', activation='sigmoid'))
    model.compile(optimser='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Flower client class definition
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self):
        # Return model parameters (weights)
        return self.model.get_weights()

    def set_parameters(self, parameters):
        # Set model parameters (weights)
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        # Update local model weights and train the model
        self.set_parameters(parameters)
        self.model.fit(xTrain, yTrain, epochs=1, batch_size=32, verbose=0)
        return self.get_parameters(), len(xTrain), {}


if __name__ == "__main__":
    fileNumber = __file__[-5:].replace(".py", "")
    if fileNumber[0].isdigit():
        clientID = int(fileNumber)
    else:
        clientID = int(fileNumber[1])


    print(f"Client: {clientID}")

    # Get dataset for local model
    # TODO: Consider making this its own helper class
    xTrain, xTest, yTrain, yTest = SVMHelper.loadDataset(clientID-1)

    # Print label distributions
    unique, counts = np.unique(yTrain, return_counts=True)
    trainCounts = dict(zip(unique, counts))
    print("Label distribution in the training set:", trainCounts)

    unique, counts = np.unique(yTest, return_counts=True)
    testCounts = dict(zip(unique, counts))
    print("Label distribution in the test set:", testCounts, "\n")

    model = CreateModel()
    client = FlowerClient(model)
    fl.client.start_numpy_client(server_address='localhost:8080', client=FlowerClient())

