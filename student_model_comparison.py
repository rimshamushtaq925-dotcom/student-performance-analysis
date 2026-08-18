import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# -----------------------------------
# 1. Load Dataset
# -----------------------------------
df = pd.read_csv("student_performance.csv")

print("Dataset:")
print(df.head())

# -----------------------------------
# 2. Data Cleaning
# -----------------------------------
df = df.drop_duplicates()
df = df.dropna()

# Create target column if it does not exist
# Pass = 1, Fail = 0
if "Pass" not in df.columns:
    df["Pass"] = (df["Final Score"] >= 60).astype(int)

# -----------------------------------
# 3. Feature Selection
# -----------------------------------
# Final Score is NOT used as a feature
# because Pass/Fail is created from Final Score.
features = [
    "Attendance",
    "Assignment Score",
    "Midterm Score"
]

X = df[features]
y = df["Pass"]

# -----------------------------------
# 4. Feature Engineering
# -----------------------------------
# Create average of pre-final performance
X = X.copy()
X["Average_Pre_Final"] = (
    X["Assignment Score"] + X["Midterm Score"]
) / 2

print("\nSelected Features:")
print(X.columns.tolist())

# -----------------------------------
# 5. Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------------
# 6. Previous Model
# -----------------------------------
previous_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])
print("\nTarget distribution:")
print(y.value_counts())

print("\nTraining target distribution:")
print(y_train.value_counts())
previous_model.fit(X_train, y_train)

previous_predictions = previous_model.predict(X_test)
previous_probabilities = previous_model.predict_proba(X_test)[:, 1]

# -----------------------------------
# 7. Hyperparameter Tuning
# -----------------------------------
tuning_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"]
}

grid_search = GridSearchCV(
    tuning_pipeline,
    param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

tuned_model = grid_search.best_estimator_

tuned_predictions = tuned_model.predict(X_test)
tuned_probabilities = tuned_model.predict_proba(X_test)[:, 1]

print("\nBest Parameters:")
print(grid_search.best_params_)

# -----------------------------------
# 8. Evaluation Function
# -----------------------------------
def evaluate_model(name, y_true, predictions, probabilities):

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_true, probabilities)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print(f"ROC-AUC:   {roc_auc:.2f}")

    return [accuracy, precision, recall, f1, roc_auc]


previous_scores = evaluate_model(
    "Previous Logistic Regression",
    y_test,
    previous_predictions,
    previous_probabilities
)

tuned_scores = evaluate_model(
    "Tuned Logistic Regression",
    y_test,
    tuned_predictions,
    tuned_probabilities
)

# -----------------------------------
# 9. Classification Report
# -----------------------------------
print("\nTuned Model Classification Report:")
print(classification_report(y_test, tuned_predictions))

# -----------------------------------
# 10. Confusion Matrix
# -----------------------------------
cm = confusion_matrix(y_test, tuned_predictions)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Tuned Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("day6_tuned_confusion_matrix.png")
plt.show()

# -----------------------------------
# 11. Model Comparison Chart
# -----------------------------------
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]

comparison_df = pd.DataFrame({
    "Metric": metrics,
    "Previous Model": previous_scores,
    "Tuned Model": tuned_scores
})

print("\nModel Comparison:")
print(comparison_df)

comparison_df.plot(
    x="Metric",
    y=["Previous Model", "Tuned Model"],
    kind="bar",
    figsize=(9, 5)
)

plt.title("Previous vs Tuned Model")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("day6_model_comparison.png")
plt.show()

# -----------------------------------
# 12. ROC Curve
# -----------------------------------
previous_fpr, previous_tpr, _ = roc_curve(
    y_test,
    previous_probabilities
)

tuned_fpr, tuned_tpr, _ = roc_curve(
    y_test,
    tuned_probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    previous_fpr,
    previous_tpr,
    label="Previous Model"
)

plt.plot(
    tuned_fpr,
    tuned_tpr,
    label="Tuned Model"
)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Previous vs Tuned Model")
plt.legend()

plt.tight_layout()
plt.savefig("day6_roc_curve.png")
plt.show()

print("\nDay 6 completed successfully!")