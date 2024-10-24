import pandas as pd
import time
import Datacleaner
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
# TODO: Consider the metrics being tracked, IF is NOT a classifier


def GetDataset(path):
    df = pd.read_csv(path)
    if df.iloc[:, -1].dtypes != 'int64':
        df = Datacleaner.LabelEncode(df)

    # Checks if it is dual class
    if df.iloc[:, -1].nunique() != 2:
        print("iForest requires binary classification.")
        return
    
    """
    Isolation forest is not a classifier:
    0 -> Not anomalous
    1 -> Anomalous
    
    The anomalous value needs to be identified so that accurate classification can occur
    Ideally, the dataset is not balanced
    """

    majorityValue = df.iloc[:,-1].mode()[0]
    

    # Prepare data for ML
    X = df.iloc[:, :-1]
    X = Datacleaner.Normalise(X)
    y = df.iloc[:, -1]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
    return [xTrain, xTest, yTrain, yTest]


def IFLabelEncode(value, dataframe):
    # This is necessary due to the method by which IF classifies data
    df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 0 if x == value else 1)
    return df


def Run(path)
    xTrain, xTest, yTrain, yTest = GetDataset(path)

    initial = time.time()
    model = IsolationForest(random_state=42)
model.fit(xTrain)
    trainingDelay = time.time()

    yPred = model.predict(xTest)
    totalTime = time.time()
    accuracy = accuracy_score(yTest, yPred)

    # Display evaluations
    print(f"Accuracy: {accuracy}")
    print(classification_report(yTest, yPred))
    print(f"Training Delay: {str(trainingDelay-initial)}")
    print(f"Total Elapsed Time: {str(totalTime-initial)}")


if __name__ == "__main__":
    path = ""
    Run(path)
    

    
