import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# TODO: Fold functions into flows


def Main(path = ""):
    # This almost certainly means this is in debug mode
    print("DEBUG MODE")
    if path == "":
        iris = datasets.load_iris()

        df = pd.DataFrame (
                iris.data,
                columns = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalHeight',])

        df['Label'] = iris.target
        return df

    return pandas.read_csv(path)


def CalculateCorrelation(dataframe):

    # TODO: Make it look presentable, especially in paper form
    matrix = dataframe.corr()
    # RdBu_r, BrBG_r, and PuOr_r look really cool
    matrix = matrix.style.background_gradient(cmap='coolwarm') \
            .set_table_styles(
                    [{'selector' : 'td',
                      'props' : [('border', '1px solid black')]}])

    matrix.to_html("Correlation.html")


def SplitDataset(dataframe, n):
    path = "C:/Users/markd/Desktop/Research/Splitdatasets"
    
    # Split into n chunks
    chunks = np.array_split(df, n)
    for i in range(n):
        # Export
        pathstring = f"{path}{n}.csv"
        chunks[i].to_csv(pathstring, index=False)


def ExtractFeatures(dataframe, featureList):
    dataframe = dataframe.drop(featureList, axis=1)
    return dataframe


# toKeep should be a list of strings (it is best if this is done on non-encoded data)
def ExtractClasses(dataframe, toKeep):
    # This is an export function without a return
    labels = []
    filteredDF = dataframe[dataframe['label'].isin(labels)]
    # NOTE: export path should include file name
    exportPath = ""
    filteredDF.to_csv(exportPath)


def Normalise(x):
    # There is no concievable reason to scale y
    scaler = StandardScaler()
    xScaled = scaler.fit_transform(x)
    return x


def LabelEncode(dataframe):
    labelEncoder = LabelEncoder()
    dataframe.iloc[:, -1] = labelEncoder.fit_transform(df.iloc[:, -1])
    return dataframe


def CrossValidate(dataset, folds): 
    # Load dataset 
    X = df.iloc[:, :-1]
    X = Normalise(X)
    y = df.iloc[:, -1]

    # Cross validate with logistic regression
    model = LogisticRegression(max_iter=1000, C=10, solver='liblinear')
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    # Display
    print(f"Mean cross validation score: {scores.mean()}")


def DisplayStats(dataframe):
    # This is to determine useful information for data processing and presentation
    print(f"Data shape: {dataframe.shape}")


# This should probably be called from an external class/file
def MergeDatasets():
    # This doesn't have or need a return as it exports data
    # TODO: Specify this
    firstPath = ""
    setPath = ""
    dfList = []

    for filename in os.listdir(firstPath):
        if filename.endswith('.csv'):
            filePath = os.path.join(firstPath, filename)
            df = pd.read_csv(filePath)
            dfList.append(df)

    mergedDF = pd.concat(dfList, ignore_index=True)
    mergedDF.to_csv(setPath, index=False)
    print("Files exported successfully.")


# Main function
if __name__ == "__main__":
    # Path needs to be entered here, no path means debug mode
    df = Main()
    CalculateCorrelation(df)
    LabelEncode(df)
    # df = ExtractFeatures(df, ['sepal length (cm)'])
    # df = Normalise(df)
    CrossValidate(df, 3)
