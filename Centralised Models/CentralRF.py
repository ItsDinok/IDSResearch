import pandas as pd
import time
import Datacleaner
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def GetDataset(path):
    df = pd.read_csv(path)
    if df.iloc[:, -1].dtypes != 'int64':
        df = Datacleaner.LabelEncode(df)

    # Prepare data for ML
    X = df.iloc[:, :-1]
    X = Datacleaner.Normalise(X)
    y = df.iloc[:, -1]
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

    return [xTrain, xTest, yTrain, yTest]


def Run(path):
    xTrain, xTest, yTrain, yTest = GetDataset(path)

    initial = time.time()
    model = RandomForestClassifier(class_weight = 'balanced',
                                   criterion = 'entropy',
                                   n_estimators = 100,
                                   max_depth = 40,
                                   min_samples_split = 2,
                                   min_samples_leaf = 1)

    model.fit(xTrain, yTrain)
    trainingDelay = time.time()

    yPred = model.predict(xTest)
    totalTime = time.time()
    accuracy = accuracy_score(yTest, yPred)
    
    # Display evaluations
    print(f"Accuracy: {accuracy}")
    print(classification_report(yTest, yPred))
    print(f"Training Delay: {str(trainingDelay-initial)}")
    print(f"Total Elapsed: {str(totalTime-initial)}")


if __name__ == "__main__":
    # This could be done with arguments, or folded into one file
    path = ""
    Run(path)

