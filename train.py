"""Trains the Horizon risk model. Run once: python train.py"""
import json, pickle
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
        "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
# In this dataset a 0 in these columns means "not measured", not a real value.
NOT_MEASURED = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df = pd.read_csv(URL, names=COLS)
df[NOT_MEASURED] = df[NOT_MEASURED].replace(0, np.nan)
df.to_csv("data/diabetes.csv", index=False)

X, y = df.drop(columns="Outcome"), df["Outcome"]
pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler()),
                 ("model", LogisticRegression(max_iter=1000))])

cv = cross_validate(pipe, X, y, scoring=["accuracy", "roc_auc", "recall"],
                    cv=StratifiedKFold(5, shuffle=True, random_state=42))
metrics = {"accuracy": float(cv["test_accuracy"].mean()),
           "auc": float(cv["test_roc_auc"].mean()),
           "recall": float(cv["test_recall"].mean()),
           "patients": int(len(df))}
pipe.fit(X, y)
pickle.dump(pipe, open("model.pkl", "wb"))
json.dump(metrics, open("metrics.json", "w"), indent=2)
print(metrics)
