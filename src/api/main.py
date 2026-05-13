# Import json so we can read model metric information from a JSON file
import json

# Import joblib so we can load the trained machine learning model
import joblib

# Import pandas so we can convert API input into a model-ready table
import pandas as pd

# Import FastAPI so we can build the backend API
from fastapi import FastAPI

# Import BaseModel and Field so we can define and validate API input
from pydantic import BaseModel, Field


# Create the FastAPI application
app = FastAPI(
    title="CampusPath AI",
    description="Student course success prediction API",
    version="1.0.0"
)


# Load the trained model once when the API starts
model = joblib.load("models/course_success_model.pkl")


# Define the exact input data the API expects
class StudentInput(BaseModel):
    # Student GPA must be between 0.0 and 4.0
    gpa: float = Field(..., ge=0.0, le=4.0)

    # Weekly study hours should be between 0 and 40
    study_hours: float = Field(..., ge=0, le=40)

    # Attendance rate must be between 0 and 100 percent
    attendance_rate: float = Field(..., ge=0, le=100)

    # Previous math grade must be between 0 and 100
    previous_math_grade: float = Field(..., ge=0, le=100)

    # Previous CS grade must be between 0 and 100
    previous_cs_grade: float = Field(..., ge=0, le=100)

    # Credit load should be between 6 and 21 credits
    credit_load: float = Field(..., ge=6, le=21)

    # Course difficulty is on a 1 to 5 scale
    course_difficulty: float = Field(..., ge=1, le=5)


# Create a home route to confirm the API is running
@app.get("/")
def home():
    return {
        "message": "CampusPath AI API is running",
        "endpoints": {
            "predict": "/predict",
            "docs": "/docs"
        }
    }

# Create a metrics route to show model information and performance
@app.get("/metrics")
def get_metrics():
    # Open the saved model metrics JSON file
    with open("models/model_metrics.json", "r") as file:
        metrics = json.load(file)

    # Return the metrics as a JSON response
    return metrics


# Helper function that turns probability into risk level
def get_risk_level(success_probability):
    # 75% or higher means the student is likely prepared
    if success_probability >= 0.75:
        return "Low Risk"

    # 50% to 74% means the student may need some support
    elif success_probability >= 0.50:
        return "Medium Risk"

    # Below 50% means the student may need strong support before taking the course
    else:
        return "High Risk"


# Helper function that creates personalized recommendations
def generate_recommendations(student, success_probability):
    # Create an empty list to store advice
    recommendations = []

    # If attendance is low, recommend improving attendance
    if student.attendance_rate < 70:
        recommendations.append("Improve attendance before or during this course.")

    # If study hours are low, recommend more weekly study time
    if student.study_hours < 8:
        recommendations.append("Increase weekly study hours to strengthen course preparation.")

    # If previous math grade is low, recommend reviewing math skills
    if student.previous_math_grade < 70:
        recommendations.append("Review math foundations related to the target course.")

    # If previous CS grade is low, recommend reviewing CS concepts
    if student.previous_cs_grade < 70:
        recommendations.append("Review prerequisite computer science concepts before taking this course.")

    # If credit load is high, recommend reducing workload
    if student.credit_load > 17:
        recommendations.append("Consider taking a lighter credit load to reduce academic pressure.")

    # If course difficulty is high, recommend support resources
    if student.course_difficulty >= 4:
        recommendations.append("Use tutoring, office hours, or study groups because this is a difficult course.")

    # If the model predicts low success probability, add a general support recommendation
    if success_probability < 0.50:
        recommendations.append("Meet with an academic advisor before registering for this course.")

    # If no specific concerns were found, give positive guidance
    if len(recommendations) == 0:
        recommendations.append("Student appears prepared based on the current academic profile.")

    # Return the final recommendation list
    return recommendations


# Create the prediction endpoint
@app.post("/predict")
def predict(student: StudentInput):

    # Convert validated API input into a dictionary
    student_dict = student.model_dump()

    # Convert the dictionary into a pandas DataFrame
    student_df = pd.DataFrame([student_dict])

    # Use the trained model to predict 0 or 1
    prediction = model.predict(student_df)[0]

    # Get probabilities for class 0 and class 1
    probabilities = model.predict_proba(student_df)[0]

    # Class 1 represents success/pass, so we grab index 1
    success_probability = probabilities[1]

    # Convert model output into readable prediction text
    if prediction == 1:
        prediction_text = "Likely to pass"
    else:
        prediction_text = "At risk of struggling"

    # Convert probability into risk level
    risk_level = get_risk_level(success_probability)

    # Generate personalized recommendations based on the input and model probability
    recommendations = generate_recommendations(student, success_probability)

    # Return the prediction response as JSON
    return {
        "prediction": prediction_text,
        "success_probability": round(success_probability * 100, 2),
        "risk_level": risk_level,
        "recommendations": recommendations
    }