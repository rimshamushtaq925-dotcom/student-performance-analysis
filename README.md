# Student Performance Analysis – Day 3

## Project Overview

This project analyzes student performance data using Python and Pandas.

The analysis includes data cleaning, basic statistics, student performance analysis, and data visualization using Matplotlib.

## Dataset

The dataset contains student information including:

- Student Name
- Age
- Gender
- Course
- Attendance
- Assignment Score
- Midterm Score
- Final Score

## Analysis Performed

The program performs the following analysis:

1. Loads the student performance dataset.
2. Displays basic information about the dataset.
3. Calculates average scores.
4. Finds the highest and lowest scores.
5. Identifies students with attendance below 75%.
6. Identifies students who are at risk of failing.
7. Calculates average final score by course.
8. Analyzes the relationship between attendance and final score.
9. Handles missing and invalid values.

## Visualizations

Three charts were created using Matplotlib:

1. Final Score Distribution
2. Average Final Score by Course
3. Attendance vs Final Score

## Files

- `student_performance.csv` – Dataset
- `student_analysis.py` – Python analysis code
- `chart1_score_distribution.png` – Score distribution chart
- `chart2_average_score_by_course.png` – Average score by course
- `chart3_attendance_vs_final_score.png` – Attendance vs final score chart

## Conclusion

The analysis shows that student performance varies across different courses. Students with lower attendance may be more likely to have lower final scores. The course-wise comparison also helps identify differences in average performance between courses.

Overall, this project demonstrates the basic data analysis workflow:

**Dataset → Pandas → Data Cleaning → Analysis → Visualization**

## Technologies Used

- Python
- Pandas
- Matplotlib