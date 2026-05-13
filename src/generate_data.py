## SECTION 1 - SYNTHETIC STUDENT DATABASE GENERATOR ##

## This part of the code randomly generates 1,000 fake student records
## The fake student record has GPA, Study Hours, Attendance Rate, Previous Math Grade
## Previous Computer Science Grade, Credit Load, and Course Difficulty

import os #os so we can work with folders and file paths
import numpy as np # NumPy for numerical operations an random data generation
import pandas as pd #pandas for creating and saving datasets in table format

# Here i set a random seed so the gnerated data is reproducible
# This means every run creates the same dataset.
np.random.seed(42)

# This is the number of fake student records we want to generate
NUM_STUDENTS = 1000

# Generate random GPA values between 1.8 and 4.0
gpa = np.round(np.random.uniform(0.0, 4.0, NUM_STUDENTS), 2)

# Generate random study hours between 1 and 25
study_hours = np.random.randint(0, 26, NUM_STUDENTS)

# Generate random attendance percentages between 50 and 100
attendance_rate = np.random.randint(0, 101, NUM_STUDENTS)

# Generate previous math grades between 50 and 100
previous_math_grade = np.random.randint(0, 101, NUM_STUDENTS)

# Generate previous computer science grades between 50 and 100
previous_cs_grade = np.random.randint(0, 101, NUM_STUDENTS)

# Generate semester credit loads between 9 and 18 credits
credit_load = np.random.randint(9, 19, NUM_STUDENTS)

# Generate course difficulty ratings between 1 and 5
course_difficulty = np.random.randint(1, 6, NUM_STUDENTS)

# Create a weighted success score.
# Higher GPA, study hours, attendance, and grades improve success.
# Higher credit loads and harder courses reduce success probability.
success_score = (
    gpa * 20
    + study_hours * 1.5
    + attendance_rate * 0.25
    + previous_math_grade * 0.15
    + previous_cs_grade * 0.2
    - credit_load * 1.2
    - course_difficulty * 5
)

# Add noise to make predictions less perfect
noise = np.random.normal(0, 15, NUM_STUDENTS)

# Add the noise to the success scores
success_score = success_score + noise

# Convert the success score into a binary label.
# Students with scores above 80 are considered successful.
passed = (success_score >= 80).astype(int)

# Create a pandas DataFrame to organize all student records into a table
df = pd.DataFrame({
    "gpa": gpa,
    "study_hours": study_hours,
    "attendance_rate": attendance_rate,
    "previous_math_grade": previous_math_grade,
    "previous_cs_grade": previous_cs_grade,
    "credit_load": credit_load,
    "course_difficulty": course_difficulty,
    "passed": passed
})

# Create the data folder if it does not already exist
os.makedirs("data", exist_ok=True)

# Save the dataset as a CSV file
df.to_csv("data/student_data.csv", index=False)

# Print confirmation that the dataset was created
print("Dataset created successfully!")

# Display the first 5 rows of the dataset
print(df.head())

# Display the dataset dimensions
print("\nDataset shape:", df.shape)

# Display how many students passed vs failed
print("\nPass/fail counts:")
print(df["passed"].value_counts())

## SECTION 1 ENDING ## 