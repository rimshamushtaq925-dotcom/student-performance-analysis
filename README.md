## Day 6 – Feature Engineering & Hyperparameter Tuning

### Objective
Improved the Student Performance Prediction model using feature engineering, feature scaling, and hyperparameter tuning.

### Tasks Performed
- Data Cleaning
- Feature Selection
- Feature Engineering
- Feature Scaling using StandardScaler
- Train-Test Split
- Hyperparameter Tuning using GridSearchCV
- Comparison of Previous and Tuned Logistic Regression models

### Model Evaluation

| Metric | Previous Model | Tuned Model |
|---|---:|---:|
| Accuracy | 0.833 | 1.000 |
| Precision | 0.833 | 1.000 |
| Recall | 1.000 | 1.000 |
| F1 Score | 0.909 | 1.000 |
| ROC-AUC | 1.000 | 1.000 |

### Day 6 Results
The tuned Logistic Regression model achieved improved Accuracy, Precision, and F1 Score compared with the previous model.

### Day 6 Visualizations
- `day6_model_comparison.png` – Previous vs Tuned model comparison
- `day6_roc_curve.png` – ROC curve comparison
- `day6_tuned_confusion_matrix.png` – Tuned model confusion matrix

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