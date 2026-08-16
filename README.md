# Student Performance - Model Comparison

## Objective

The objective of this project is to compare two machine learning classification algorithms for predicting whether a student will pass or fail based on their academic performance.

## Dataset

The dataset contains 30 student records with information about:

- Attendance
- Assignment Score
- Midterm Score
- Final Score

The target variable is:

- 1 = Pass
- 0 = Fail

## Machine Learning Models

Two classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree

## Train-Test Split

The dataset was divided into:

- Training records: 24
- Testing records: 6

The test size was 20%.

## Model Accuracy

| Model | Accuracy |
|---|---:|
| Logistic Regression | 83.33% |
| Decision Tree | 83.33% |

## Classification Results

Both models produced the same overall performance on the test dataset.

### Logistic Regression

- Accuracy: 83.33%
- Precision: 0.88
- Recall: 0.83
- F1-score: 0.83

### Decision Tree

- Accuracy: 83.33%
- Precision: 0.88
- Recall: 0.83
- F1-score: 0.83

## Confusion Matrix

Both models produced the following confusion matrix:

| Actual / Predicted | Fail (0) | Pass (1) |
|---|---:|---:|
| Fail (0) | 3 | 0 |
| Pass (1) | 1 | 2 |

This means that 5 out of 6 test records were correctly classified.

## Conclusion

Logistic Regression and Decision Tree achieved the same accuracy of 83.33% on the test dataset. Both models also produced the same precision, recall, and F1-score.

Therefore, neither model has a clear performance advantage on this dataset.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Project Files

- `student_model_comparison.py` - Model training and comparison
- `student_performance.csv` - Student dataset
- `student_analysis.py` - Student performance analysis
- `student_prediction.py` - Student prediction model
- `confusion_matrix.png` - Confusion matrix visualization
- `README.md` - Project documentation