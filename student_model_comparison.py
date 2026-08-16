import pandas as pd
# Load dataset
df = pd.read_csv("student_performance.csv")
# Display first 5 rows
print("First 5 rows:")
print(df.head())
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())
# Remove rows with missing values
df = df.dropna()
print("\nDataset after cleaning:")
print(df.head())
print("\nTotal records:", len(df))
from sklearn.model_selection import train_test_split
# Features
X = df[["Attendance", "Assignment Score", "Midterm Score"]]
# Target
# Create Pass/Fail target
df["Pass"] = (df["Final Score"] >= 80).astype(int)
print("\nPass distribution:")
print(df["Pass"].value_counts())
print("\nFinal Score range:")
print(df["Final Score"].min(), "to", df["Final Score"].max())
print("\nFinal Scores:")
print(df["Final Score"].sort_values().tolist())
y = df["Pass"]
print("\nColumn names:")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
# Train Model 1 - Logistic Regression
from sklearn.linear_model import LogisticRegression
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
print("\nLogistic Regression model trained successfully!")
# Train Model 2 - Decision Tree
from sklearn.tree import DecisionTreeClassifier
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
print("\nDecision Tree model trained successfully!")
# Predictions
logistic_predictions = logistic_model.predict(X_test)
decision_tree_predictions = decision_tree_model.predict(X_test)
from sklearn.metrics import accuracy_score
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
print("\nModel Accuracy:")
print("Logistic Regression:", logistic_accuracy)
print("Decision Tree:", decision_tree_accuracy)
from sklearn.metrics import classification_report
print("\nLogistic Regression Report:")
print(classification_report(y_test, logistic_predictions))
print("\nDecision Tree Report:")
print(classification_report(y_test, decision_tree_predictions))
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Confusion Matrix - Logistic Regression
logistic_cm = confusion_matrix(y_test, logistic_predictions)

plt.figure(figsize=(5, 4))
sns.heatmap(logistic_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Confusion Matrix - Decision Tree
decision_tree_cm = confusion_matrix(y_test, decision_tree_predictions)

plt.figure(figsize=(5, 4))
sns.heatmap(decision_tree_cm, annot=True, fmt="d", cmap="Greens")
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()