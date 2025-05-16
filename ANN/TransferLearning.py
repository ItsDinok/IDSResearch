import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def PrepareData(data, balance=True):
    # Data is a pandas dataframe
    le = LabelEncoder()
    data.iloc[:, -1] = le.fit_transform(data.iloc[:, -1])
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(int)

    # Create model
    model = RandomForestClassifier(random_state=42, n_estimators=10)
    model.fit(x, y)

    leafIndices = np.array([tree.apply(x) for tree in model.estimators_])

    # Get the gini per sample
    giniPerSample = []
    
    for treeIDX, tree in enumerate(model.estimators_):
        impurity = tree.tree_.impurity
        sampleGini = impurity[leafIndices[treeIDX]]
        giniPerSample.append(sampleGini)

    giniPerSample = np.array(giniPerSample).T
    meanGiniPerSample = giniPerSample.mean(axis=1)
    
    # n is the number of samples
    n = 1000
    perClass = n // len(np.unique(y))

    if balance:
        bestSampleIndices = np.argsort(meanGiniPerSample)[:n]
        bestSamples = data.iloc[bestSampleIndices]
    else:
        for cls in np.unique(y):
            classIndices = data[y == cls].index
            classGini = meanGiniPerSample[classIndices]
            bestClassIndices = classIndices[np.argsort(classGini)[:perClass]]
            bestSamples = pd.concat([bestSamples, data.loc[bestClassIndices]])
    
    print(bestSamples)
    # bestSamples = CorrelateReduction(bestSamples)
    return bestSamples

def CorrelateReduction(data):
    correlation = data.corr()['label']
    relevantColumns = correlation[abs(correlation) >= 0.1].index.tolist()
    return data[relevantColumns]

def Main():
    df = pd.read_csv("C:/Users/markd/Desktop/Research/Utility/ClassExtractedData.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    print(PrepareData(df))
    

if __name__ == "__main__":
    Main()

