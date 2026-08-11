import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
df = pd.read_csv("student_performance.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
df["Pass"] = (df["Final Score"] >=60).astype(int)
print(df[["Final Score","Pass"]])
x = df[["Attendance" , "Assignment Score" , "Midterm Score" , "Final Score"]]
y = df["Pass"]
print(x.head())
print(y.head())
print(df.columns.tolist())
# split data into training and testing sets
x_train ,x_test ,y_train ,y_test =train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)
# Create the Logistic Regression model
model = LogisticRegression()
# Train the model
print("Minimum Final Score:" ,df["Final Score"].min())
print("Pass/Failcount:")
print(df["Pass"].value_counts())
model.fit(x_train, y_train)
print("Model training completed!")
# Make predictions on testing data
y_pred=model.predict(x_test)
print("Predictions:")
print(y_pred)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Fail", "Pass"])
plt.yticks([0, 1], ["Fail", "Pass"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.savefig("confusion_matrix.png")
plt.show()
# Actual vs Predicted Visualization
plt.figure(figsize=(7, 4))
plt.plot(range(len(y_test)), y_test.values, marker="o", label="Actual")
plt.plot(range(len(y_pred)), y_pred, marker="x", label="Predicted")
plt.title("Actual vs Predicted Pass/Fail")
plt.xlabel("Test Student")
plt.ylabel("Pass/Fail")
plt.yticks([0, 1], ["Fail", "Pass"])
plt.legend()
plt.savefig("actual_vs_predicted.png")
plt.show()
