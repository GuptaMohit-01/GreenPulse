# train_model.py
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             classification_report, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42
N_JOBS = -1

# ------------ 1) Load data ------------
# Expecting Kaggle Crop Recommendation dataset
# Columns: N,P,K,temperature,humidity,ph,rainfall,label
DATA_PATH = Path("data/crop_recommendation.csv")
assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH.resolve()}"

df = pd.read_csv(DATA_PATH)

# Basic sanity checks
required_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"]
missing = [c for c in required_cols if c not in df.columns]
assert not missing, f"Missing columns in dataset: {missing}"

# ------------ 2) Features & target ------------
feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
target_col = "label"

X = df[feature_cols].copy()
y = df[target_col].astype("category")

# ------------ 3) Train/Validation split ------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ------------ 4) Preprocess + Model pipeline ------------
# All numeric -> scale only temp/humidity/ph/rainfall (tree models don't need scaling,
# but scaling helps if you later switch to linear/SVM models)
scale_cols = ["temperature", "humidity", "ph", "rainfall"]
passthrough_cols = ["N", "P", "K"]

preprocessor = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), scale_cols),
        ("keep", "passthrough", passthrough_cols),
    ],
    remainder="drop",
)

# Base RandomForest (robust baseline)
rf = RandomForestClassifier(
    random_state=RANDOM_STATE,
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",   # handle any minor imbalance
    n_jobs=N_JOBS
)

# Full pipeline (preprocess -> RF)
base_pipe = Pipeline(steps=[
    ("prep", preprocessor),
    ("rf", rf),
])

# ------------ 5) Hyperparameter tuning via CV ------------
param_grid = {
    "rf__n_estimators": [300, 500],
    "rf__max_depth": [None, 10, 20],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
    "rf__max_features": ["sqrt", "log2"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

search = GridSearchCV(
    estimator=base_pipe,
    param_grid=param_grid,
    scoring="f1_macro",
    n_jobs=N_JOBS,
    cv=cv,
    verbose=1,
    refit=True,
)

search.fit(X_train, y_train)

print("\nBest params:", search.best_params_)
print("Best CV macro-F1:", round(search.best_score_, 4))

best_pipe = search.best_estimator_

# ------------ 6) Probability calibration ------------
# RF probabilities can be overconfident/underconfident. Calibrate on validation split.
try:
    calibrated = CalibratedClassifierCV(
        estimator=best_pipe,
        method="isotonic",
        cv="prefit"
    )
except TypeError:
    # For older sklearn
    calibrated = CalibratedClassifierCV(
        base_estimator=best_pipe,
        method="isotonic",
        cv="prefit"
    )

calibrated.fit(X_valid, y_valid)

# ------------ 7) Evaluate ------------
def evaluate(model, X, y, average="macro"):
    proba = model.predict_proba(X)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average=average)
    # multi-class ROC-AUC macro
    # Need one-vs-rest probabilities
    try:
        classes = model.classes_
        # map labels to indices
        y_bin = pd.get_dummies(pd.Categorical(y, categories=classes))
        auc = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        auc = np.nan
    ll = log_loss(y, proba)
    return acc, f1, auc, ll, preds, proba

acc_tr, f1_tr, auc_tr, ll_tr, preds_tr, proba_tr = evaluate(calibrated, X_train, y_train)
acc_va, f1_va, auc_va, ll_va, preds_va, proba_va = evaluate(calibrated, X_valid, y_valid)

print("\n=== TRAIN METRICS ===")
print(f"Accuracy: {acc_tr:.4f} | F1(macro): {f1_tr:.4f} | AUC(macro): {auc_tr:.4f} | LogLoss: {ll_tr:.4f}")

print("\n=== VALID METRICS ===")
print(f"Accuracy: {acc_va:.4f} | F1(macro): {f1_va:.4f} | AUC(macro): {auc_va:.4f} | LogLoss: {ll_va:.4f}")

print("\nClassification Report (Validation):")
print(classification_report(y_valid, preds_va))

print("\nConfusion Matrix (Validation):")
print(confusion_matrix(y_valid, preds_va))

# ------------ 8) Persist artifacts ------------
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
LABELS_PATH = ARTIFACTS_DIR / "label_classes.json"
META_PATH = ARTIFACTS_DIR / "meta.json"

# Save calibrated model
joblib.dump(calibrated, MODEL_PATH)

# Save classes (order matters)
classes_list = list(calibrated.classes_)
with open(LABELS_PATH, "w", encoding="utf-8") as f:
    json.dump(classes_list, f, ensure_ascii=False, indent=2)

# Save meta (for app to validate feature order etc.)
meta = {
    "feature_order": feature_cols,
    "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
    "valid_shape": [int(X_valid.shape[0]), int(X_valid.shape[1])],
    "metrics": {
        "train": {"accuracy": acc_tr, "f1_macro": f1_tr, "auc_macro": float(auc_tr) if auc_tr==auc_tr else None, "log_loss": ll_tr},
        "valid": {"accuracy": acc_va, "f1_macro": f1_va, "auc_macro": float(auc_va) if auc_va==auc_va else None, "log_loss": ll_va},
    },
    "best_params": search.best_params_,
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "random_state": RANDOM_STATE
}
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\nSaved: {MODEL_PATH}")
print(f"Saved: {LABELS_PATH}")
print(f"Saved: {META_PATH}")

# --- Evaluate Accuracy Only ---
from sklearn.metrics import accuracy_score

train_preds = calibrated.predict(X_train)
valid_preds = calibrated.predict(X_valid)

train_acc = accuracy_score(y_train, train_preds)
valid_acc = accuracy_score(y_valid, valid_preds)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {valid_acc:.4f}")

