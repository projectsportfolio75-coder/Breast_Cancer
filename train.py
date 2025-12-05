# train_model.ipynb (copy–paste into Jupyter)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("breast_cancer_dataframe.csv")   # change file name if needed

# X, y
X = df.drop("target", axis=1)
y = df["target"]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save model + scaler in one dict
saved_model = {"model": model, "scaler": scaler, "features": X.columns.tolist()}

# Export clean pkl
with open("model2.pkl", "wb") as f:
    pickle.dump(saved_model, f)

print("model.pkl saved successfully!")
