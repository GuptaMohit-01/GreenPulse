import os
from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

ROOT = Path(__file__).resolve().parent

# -------- Locate CSV --------
csv_candidates = [
    ROOT / "data" / "crop_recommendation.csv",
    ROOT / "data" / "Crop_recommendation.csv",
    ROOT / "data" / "Crop Recommendation.csv",
]
csv_path = None
for c in csv_candidates:
    if c.exists():
        csv_path = c
        break

if csv_path is None:
    print("❌ Dataset not found. Place crop_recommendation.csv in the data folder.")
    sys.exit(1)

print(f"✅ Dataset found at: {csv_path}")
df = pd.read_csv(csv_path)

# -------- Preprocess --------
df.columns = [c.strip().lower() for c in df.columns]
df = df.rename(columns={"crop": "label"}) if "crop" in df.columns else df

required_cols = ["n", "p", "k", "temperature", "humidity", "ph", "rainfall", "label"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f"❌ Missing columns: {missing}")
    sys.exit(1)

X = df[["n", "p", "k", "temperature", "humidity", "ph", "rainfall"]]
y = df["label"]

# -------- Split --------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError as e:
    print("⚠️ Stratify failed, using normal split:", e)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# -------- Train --------
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

# -------- Evaluate --------
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc*100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# -------- Confusion Matrix --------
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
classes = sorted(y.unique())

docs_dir = ROOT / "docs"
docs_dir.mkdir(exist_ok=True)

plt.figure(figsize=(12, 9))
try:
    import seaborn as sns
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
except ImportError:
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(classes)), classes, rotation=90)
    plt.yticks(range(len(classes)), classes)
    plt.colorbar()
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(docs_dir / "evaluation.png")
plt.close()
print(f"📊 Confusion matrix saved to: {docs_dir / 'evaluation.png'}")

# -------- Save model --------
pickle_path = ROOT / "model.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(rf, f)
print(f"💾 Model saved as: {pickle_path}")
