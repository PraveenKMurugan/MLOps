import pandas as pd
from sklearn.model_selection import train_test_split
import os
 
# Define the path to the dataset
DATA_PATH = 'tourism_project/data/tourism.csv'

# Load the dataset
df = pd.read_csv(DATA_PATH)

# Separate features (X) and target (y)
X = df.drop(columns=['CustomerID', 'ProdTaken'])
y = df['ProdTaken']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save the split datasets locally as CSV files
X_train.to_csv('Xtrain.csv', index=False)
X_test.to_csv('Xtest.csv', index=False)
y_train.to_csv('ytrain.csv', index=False)
y_test.to_csv('ytest.csv', index=False)

print("Data preparation complete. Xtrain.csv, Xtest.csv, ytrain.csv, and ytest.csv saved.")
