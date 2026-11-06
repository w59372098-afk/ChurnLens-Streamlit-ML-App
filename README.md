# Project 04 — First Machine Learning Classifier

Train a Scikit-Learn Decision Tree classifier to predict a customer's churn label from account/service features.

## Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

## Pipeline
1. Load the labelled CSV.
2. Split features/target and create an 80/20 stratified train/test split.
3. Scale numeric columns and one-hot encode categorical columns.
4. Fit a Decision Tree inside a Scikit-Learn Pipeline.
5. Evaluate with accuracy, confusion matrix and classification report.
6. Save the reusable model with joblib.

## Files created
- `models/churn_decision_tree.joblib`
- `reports/metrics.json`
- `reports/confusion_matrix.png`

## Important
This is a synthetic educational dataset. Accuracy is only a learning metric, not proof that the model is suitable for real customer decisions. Review precision/recall and test on representative real data before any real use.
