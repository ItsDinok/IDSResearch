from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import sys

def SplitDataset(dataframe, n):
    path = ""

    chunks = np.array_split(dataframe, n)
    for i in range(n):
        print(i)
        pathstring = f"{path}{i}.csv"
        chunks[i].to_csv(pathstring, index=False)


df = pd.read_csv("C:/Users/markd/Desktop/FL Tools/cleaned.csv")
df = df.drop(columns=['Tot sum'])
df = df.sample(frac=1)
labelEncoder = LabelEncoder()
df.iloc[:, -1] = labelEncoder.fit_transform(df.iloc[:, -1])

if len(sys.argv) > 1:
    Datacleaner.SplitDataset(df, int(sys.argv[1]))
else:
    Datacleaner.SplitDataset(df, 2)
