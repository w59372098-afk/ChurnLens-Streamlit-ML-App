"""Train and evaluate a Decision Tree customer churn classifier."""
from pathlib import Path
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent
df = pd.read_csv(ROOT / "data" / "customer_churn.csv")
target = "churn"
features = ["tenure_months", "monthly_charges", "support_tickets",
            "contract_type", "payment_method", "internet_service"]
X, y = df[features], df[target]
numeric = ["tenure_months", "monthly_charges", "support_tickets"]
categorical = ["contract_type", "payment_method", "internet_service"]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])
model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", DecisionTreeClassifier(max_depth=5, min_samples_leaf=8,
                                           class_weight="balanced", random_state=42))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

(ROOT/"models").mkdir(exist_ok=True)
(ROOT/"reports").mkdir(exist_ok=True)
joblib.dump(model, ROOT/"models"/"churn_decision_tree.joblib")
report = classification_report(y_test, pred, output_dict=True, zero_division=0)
metrics = {"accuracy": accuracy, "confusion_matrix": cm.tolist(),
           "classification_report": report, "train_rows": len(X_train),
           "test_rows": len(X_test)}
(ROOT/"reports"/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Stay", "Churn"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Customer Churn — Confusion Matrix")
plt.tight_layout()
plt.savefig(ROOT/"reports"/"confusion_matrix.png", dpi=150)
plt.close()
print(f"Accuracy: {accuracy:.3f}")
print("Confusion matrix (rows=actual, columns=predicted):")
print(cm)
print("Saved model to models/churn_decision_tree.joblib")
