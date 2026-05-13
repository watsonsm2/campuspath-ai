# CampusPath AI

An end-to-end machine learning system for predicting student course success and academic risk.

## Project Overview

CampusPath AI is a machine learning engineering project designed to predict whether a student is likely to succeed in a target course based on academic history, study behaviors, and course difficulty.

The system simulates a real-world educational analytics platform by:

* Generating realistic student data
* Cleaning messy datasets
* Training a machine learning model
* Serving predictions through a FastAPI backend
* Returning risk analysis and personalized recommendations

This project focuses on the full ML engineering pipeline rather than only model training.

---

# Features

## Machine Learning Pipeline

* Synthetic student dataset generation
* Intentionally messy data simulation
* Data cleaning and preprocessing pipeline
* Random Forest classification model
* Model evaluation metrics
* Saved model inference pipeline

## Prediction System

* Student success prediction
* Academic risk classification
* Success probability scoring
* Personalized academic recommendations

## FastAPI Backend

* REST API prediction endpoint
* Input validation using Pydantic
* JSON prediction responses
* Interactive Swagger documentation

---

# Tech Stack

## Backend & API

* Python
* FastAPI
* Uvicorn
* Pydantic

## Machine Learning

* scikit-learn
* RandomForestClassifier
* pandas
* NumPy
* joblib

## Development Tools

* Git
* GitHub
* VS Code
* Virtual Environments

---

# Project Structure

```text
campuspath-ai/
│
├── data/
│   └── student_data.csv
│
├── models/
│   └── course_success_model.pkl
│
├── notebooks/
│
├── src/
│   ├── generate_data.py
│   ├── train_model.py
│   ├── predict.py
│   │
│   └── api/
│       └── main.py
│
├── requirements.txt
├── README.md
└── venv/
```

---

# Machine Learning Workflow

```text
Synthetic Data Generation
↓
Messy Data Simulation
↓
Data Cleaning & Validation
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
Model Saving
↓
Inference Pipeline
↓
FastAPI Model Serving
```

---

# Dataset Features

The model currently uses the following features:

| Feature             | Description                           |
| ------------------- | ------------------------------------- |
| gpa                 | Student GPA                           |
| study_hours         | Weekly study hours                    |
| attendance_rate     | Attendance percentage                 |
| previous_math_grade | Previous math course performance      |
| previous_cs_grade   | Previous computer science performance |
| credit_load         | Semester credit load                  |
| course_difficulty   | Difficulty of target course           |

## Target Label

| Label | Meaning                |
| ----- | ---------------------- |
| 1     | Likely to pass/succeed |
| 0     | At risk of struggling  |

---

# Data Cleaning Pipeline

The preprocessing pipeline handles:

* Missing values
* Duplicate rows
* Invalid numeric values
* Impossible GPA values
* Attendance outliers
* Incorrect data types
* Noisy data

---

# Model Information

## Current Model

Random Forest Classifier

## Why Random Forest?

Random Forest was selected because it:

* Performs well on structured/tabular data
* Handles nonlinear relationships
* Reduces overfitting through ensemble learning
* Works well with mixed feature interactions
* Provides strong baseline performance

---

# API Endpoints

## Home Route

```http
GET /
```

Returns API status information.

## Prediction Route

```http
POST /predict
```

### Example Request

```json
{
  "gpa": 3.2,
  "study_hours": 12,
  "attendance_rate": 88,
  "previous_math_grade": 80,
  "previous_cs_grade": 85,
  "credit_load": 15,
  "course_difficulty": 4
}
```

### Example Response

```json
{
  "prediction": "Likely to pass",
  "success_probability": 84.0,
  "risk_level": "Low Risk",
  "recommendations": [
    "Student appears prepared based on the current academic profile."
  ]
}
```

---

# Running The Project

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/campuspath-ai.git
cd campuspath-ai
```

## 2. Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Generate Dataset

```bash
python src/generate_data.py
```

---

# Train The Model

```bash
python src/train_model.py
```

---

# Run Inference Script

```bash
python src/predict.py
```

---

# Start FastAPI Server

```bash
uvicorn src.api.main:app --reload
```

---

# Open Interactive API Docs

Visit:

```text
http://127.0.0.1:8000/docs
```

---

# Future Improvements

Planned future improvements include:

* React frontend dashboard
* PostgreSQL database integration
* Docker containerization
* Cloud deployment
* Model monitoring
* Authentication system
* Student analytics dashboard
* Multi-model comparison
* Deep learning model experimentation
* Real-time inference logging

---

# Machine Learning Concepts Demonstrated

This project demonstrates:

* Supervised learning
* Binary classification
* Ensemble learning
* Data preprocessing
* Model evaluation
* Inference pipelines
* API development
* Model serving
* Deployment preparation
* ML engineering workflow

---

## Live API

The deployed CampusPath AI API is available at:

https://campuspath-ai.onrender.com

Interactive API documentation:

https://campuspath-ai.onrender.com/docs

# Author

Shawn Watson

Computer Science and Mathematics Student
Virginia Commonwealth University
