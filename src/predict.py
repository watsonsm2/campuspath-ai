# Import joblib so we can load the trained model saved from train_model.py
import joblib

# Import pandas so we can format new student data like the training dataset
import pandas as pd


# Load the trained Random Forest model from the models folder.
# This lets us use the model without training it again.
model = joblib.load("models/course_success_model.pkl")


# Create a new student example.
# This represents ONE student trying to take ONE target course.
new_student = {
    "gpa": 3.2,
    "study_hours": 12,
    "attendance_rate": 88,
    "previous_math_grade": 80,
    "previous_cs_grade": 85,
    "credit_load": 15,
    "course_difficulty": 4
}


# Convert the dictionary into a pandas DataFrame.
# The model expects tabular data with the same columns used during training.
student_df = pd.DataFrame([new_student])


# Use the trained model to predict whether the student will pass.
# Output will be 0 or 1.
prediction = model.predict(student_df)[0]


# Get prediction probabilities.
# This tells us how confident the model is for each class.
probabilities = model.predict_proba(student_df)[0]


# The probability of class 1 means probability of passing.
success_probability = probabilities[1]


# Convert the prediction into a human-readable result.
if prediction == 1:
    result = "Likely to pass"
else:
    result = "At risk of struggling"


# Convert probability into a risk level.
if success_probability >= 0.75:
    risk_level = "Low Risk"
elif success_probability >= 0.50:
    risk_level = "Medium Risk"
else:
    risk_level = "High Risk"


# Print the input student profile.
print("Student Profile:")
print(student_df)


# Print the model's prediction.
print("\nPrediction:", result)


# Print the success probability as a percentage.
print("Success Probability:", round(success_probability * 100, 2), "%")


# Print the risk level.
print("Risk Level:", risk_level)