## SECTION 2 DATA CLEANING AND MODEL TRAINING #
import os
# save our trained machine learning model
import joblib

import pandas as pd

# Import train_test_split so we can split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import RandomForestClassifier as our first machine learning model
from sklearn.ensemble import RandomForestClassifier
# Import evaluation metrics so we can measure how well our model performs
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the messy student dataset from the data folder
df = pd.read_csv("data/student_data.csv")


# Print the original dataset shape so we know how much data we started with
print("Original dataset shape:", df.shape)


# Remove duplicate rows because duplicated data can make the model biased
df = df.drop_duplicates()


# Print the dataset shape after removing duplicates
print("After removing duplicates:", df.shape)


# Convert numeric columns to actual numbers.
# Some messy values may be stored as text, like "missing".
# errors="coerce" turns bad values into NaN so we can clean them.
numeric_columns = [
    "gpa",
    "study_hours",
    "attendance_rate",
    "previous_math_grade",
    "previous_cs_grade",
    "credit_load",
    "course_difficulty",
    "passed"
]


# Loop through each numeric column and force it to become numeric
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# Remove rows where the target label is missing.
# We cannot train a supervised learning model without the correct answer.
df = df.dropna(subset=["passed"])


# Fix impossible GPA values.
# A valid GPA should be between 0.0 and 4.0.
df.loc[df["gpa"] < 0, "gpa"] = None
df.loc[df["gpa"] > 4.0, "gpa"] = None


# Fix impossible study hour values.
# Study hours should not be negative.
df.loc[df["study_hours"] < 0, "study_hours"] = None


# Cap extreme study hour values.
# More than 40 study hours for one course is unrealistic for our project.
df.loc[df["study_hours"] > 40, "study_hours"] = 40


# Fix impossible attendance values.
# Attendance should be between 0 and 100 percent.
df.loc[df["attendance_rate"] < 0, "attendance_rate"] = None
df.loc[df["attendance_rate"] > 100, "attendance_rate"] = None


# Fix impossible math grades.
# Grades should be between 0 and 100.
df.loc[df["previous_math_grade"] < 0, "previous_math_grade"] = None
df.loc[df["previous_math_grade"] > 100, "previous_math_grade"] = None


# Fix impossible computer science grades.
# Grades should be between 0 and 100.
df.loc[df["previous_cs_grade"] < 0, "previous_cs_grade"] = None
df.loc[df["previous_cs_grade"] > 100, "previous_cs_grade"] = None


# Fix impossible credit loads.
# Most full-time students take around 12 to 18 credits.
# We allow 6 to 21 to be flexible.
df.loc[df["credit_load"] < 6, "credit_load"] = None
df.loc[df["credit_load"] > 21, "credit_load"] = None


# Fix impossible course difficulty values.
# Our scale should be 1 to 5.
df.loc[df["course_difficulty"] < 1, "course_difficulty"] = None
df.loc[df["course_difficulty"] > 5, "course_difficulty"] = None


# Fill missing values using the median of each column.
# Median is often safer than mean because outliers affect it less.
for column in numeric_columns:
    if column != "passed":
        df[column] = df[column].fillna(df[column].median())


# Make sure passed is an integer.
# The model expects class labels like 0 and 1.
df["passed"] = df["passed"].astype(int)


# Print cleaned dataset information
print("\nCleaned dataset shape:", df.shape)


# Show how many missing values remain after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())


# Separate features from the target label.
# X contains the information the model uses to make predictions.
X = df.drop("passed", axis=1)


# y contains the answer the model is trying to predict.
y = df["passed"]


# Split data into training and testing sets.
# Training data teaches the model.
# Testing data checks if the model can generalize to new examples.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create the Random Forest model.
# n_estimators=100 means the forest will use 100 decision trees.
# random_state=42 makes results reproducible.
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Train the model using the training data
model.fit(X_train, y_train)


# Use the trained model to make predictions on the test data
y_pred = model.predict(X_test)


# Calculate accuracy.
# Accuracy means the percentage of predictions the model got correct.
accuracy = accuracy_score(y_test, y_pred)


# Print the model accuracy
print("\nModel Accuracy:", accuracy)


# Print the confusion matrix.
# This shows correct and incorrect predictions for each class.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Print the full classification report.
# This includes precision, recall, and F1-score.
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Create the models folder if it does not already exist
os.makedirs("models", exist_ok=True)


# Save the trained model so we can use it later in our API
joblib.dump(model, "models/course_success_model.pkl")


# Print confirmation that the model was saved
print("\nModel saved to models/course_success_model.pkl")