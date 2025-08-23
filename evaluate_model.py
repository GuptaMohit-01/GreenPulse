import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load artifacts
MODEL_PATH = Path("artifacts/model.pkl")
DATA_PATH = Path("data/crop_recommendation.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
target = "label"

X = df[features]
y = df[target]

# --- Predictions
y_pred = model.predict(X)

# --- Accuracy
acc = accuracy_score(y, y_pred)
print(f"Overall Accuracy: {acc:.4f}")

# --- Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# --- Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=model.classes_)
cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)

plt.figure(figsize=(14, 12))
sns.heatmap(cm_df, annot=False, cmap="Greens")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
plt.savefig("confusion_matrix.png")