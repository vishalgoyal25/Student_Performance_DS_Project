"# Student_Performance_DS_Project" 
🎯 Student Exam Performance Predictor

This project predicts a student's Math score based on other exam-related features. The goal is to help educators and students understand performance patterns and identify areas of improvement.

🔹 What we are doing:

We have a dataset containing students’ exam details:

Gender

Race/Ethnicity

Parental Level of Education

Lunch Type

Test Preparation Course

Writing Score

Reading Score

Using this data, we train machine learning models to predict the Math score of a student.

Steps involved:

Data Ingestion – Reading the CSV dataset and splitting into training and test sets.

Data Transformation – Preprocessing categorical and numerical features:

Encode categorical variables

Scale numerical variables

Combine features and target variable for model training

Model Training – Testing multiple regression models like:

Random Forest Regressor

Decision Tree Regressor

Gradient Boosting

Linear Regression

XGBoost

CatBoost

AdaBoost

The best model is selected based on R² score.

Once trained, the model can predict the math score for a new student given their features.

💡 Why this project:

Helps in predicting student performance without manual calculation.

Useful for teachers, schools, and students to quickly identify performance trends.

Shows an end-to-end ML workflow:

Data preprocessing → Model training → Evaluation → Prediction
