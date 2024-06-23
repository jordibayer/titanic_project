# Titanic Survival Prediction Project

This project is a monolithic application that predicts the survival of passengers on the Titanic. It consists of three main parts: Backend, Frontend, and Model.

## Table of Contents

1. [Backend](#backend)
2. [Frontend](#frontend)
3. [Model](#model)
4. [Setup and Installation](#setup-and-installation)
5. [Usage](#usage)

## Backend

The backend is built using FastAPI and provides the API endpoints for the Titanic survival prediction model.

### Technologies Used

- Python
- FastAPI
- Poetry (for dependency management)
- Joblib
- Pandas
- Scikit-learn
- XGBoost

### Setup

1. Ensure you have Python installed.
2. Install Poetry: `pip install poetry`
3. Navigate to the backend directory.
4. Install dependencies: `poetry install`

## Frontend

The frontend is a Next.js application that provides a user interface for interacting with the Titanic survival prediction model.

### Technologies Used

- Next.js
- React
- TypeScript
- Tailwind CSS
- React Hook Form

### Setup

1. Ensure you have Node.js installed.
2. Navigate to the frontend directory.
3. Install dependencies: `npm install`

## Model

The model component includes exploratory data analysis and the machine learning model for predicting Titanic survival.

### Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

### Model Development

The model development process includes:

1. Exploratory Data Analysis
2. Data Preprocessing
3. Model Selection and Training
4. Model Evaluation

## Setup and Installation

1. Clone the repository:

   ```
   cd model
   ```

2. Set up the backend:

   ```
   cd backend
   poetry install
   ```

3. Set up the frontend:

   ```
   cd ../frontend
   npm install
   ```

4. Set up the model environment (if separate from backend):
   ```
   cd ../model
   pip install -r requirements.txt
   ```

## Usage

1. Start the backend server:

   ```
   cd backend
   poetry shell
   fastapi dev backend.py
   ```

2. Start the frontend development server:

   ```
   cd frontend
   npm run dev
   ```

3. Access the application at `http://localhost:3000`
