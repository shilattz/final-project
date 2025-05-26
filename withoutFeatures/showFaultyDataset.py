import pandas as pd

df = pd.read_csv('withoutFeatures/full_faulty_dataset.csv')

print("Table format:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())