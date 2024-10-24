import pandas as pd
import time
import Datacleaner
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def GetDataset(path):
    df = pd.read_csv(path)
    if df.iloc[:, -1].dtypes != 'int64':
        Datacleaner.LabelEncode(df)


    # Prepare the data for ML
    X = df.iloc[:, :-1]
    X = Datacleaner.Normalise(X)
    y = df.iloc[:, -1]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    return [xTrain, xTest, yTrain, yTest]


def Run(path):
    xTrain, yTrain, xTest, yTest = GetDataset(path)

    initial = time.time()
    model = LinearSVC(random_state=42, tol=1e-5)
    model.fit(xTrain, yTrain)
    trainingDelay = time.time()

    yPred = model.predict(xTest)
    totalTime = time.time()
    accuracy = accuracy_score(yTest, yPred)
    print(f"Accuracy: {accuracy}")
    print(classification_report(yTest, yPred))
    print(f"Training Delay: {str(trainingDelay-initial)}s")
    print(f"Total Elapsed: {str(totalTime-initial)}s")

   
if __name__ == "__main__":
    # This could be done with arguments
    path = ""
    Run(path)
