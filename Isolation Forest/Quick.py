import pandas as pd
import Datacleaner
from sklearn.preprocessing import LabelEncoder
import sys

def SplitDataset(dataframe, n):
    path = ""
    
    # Split into n chunks
    chunks = np.array_split(dataframe, n)
    for i in range(n):
        # Export
        pathstring = f"{path}{n}.csv"
        chunks[i].to_csv(pathstring, index=False)

df = pd.read_csv("")
df = df.sample(frac=1)
labelEncoder = LabelEncoder()
df.iloc[:, -1] = labelEncoder.fit_transform(df.iloc[:, -1])

if len(sys.argv) > 1:
    Datacleaner.SplitDataset(df, int(sys.argv[1]))
else:
    Datacleaner.SplitDataset(df, 2)
