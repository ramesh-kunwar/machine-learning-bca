# Titanic Survival Predictor

This is a full-stack machine learning application that predicts whether a passenger would have survived the Titanic disaster based on various features.

## Features

- Machine learning model with >90% accuracy
- Cross-validation to prevent overfitting
- Modern, responsive web interface
- Real-time predictions
- Probability visualization

## Setup and Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

## Running the Application

1. Start the backend server (in one terminal):
```bash
python app.py
```

2. Start the frontend server (in another terminal):
```bash
python serve.py
```

3. Open your web browser and navigate to:
```
http://localhost:3000
```

## Usage

1. Fill in the passenger details in the form:
   - Passenger Class (1st, 2nd, or 3rd)
   - Gender
   - Age
   - Fare (in pounds)
   - Port of Embarkation
   - Family Size

2. Click "Predict Survival" to get the prediction

3. The result will show:
   - Whether the passenger would have survived
   - The probability of survival
   - A visual representation of the probability

## Model Details

- Algorithm: Random Forest Classifier
- Features used:
  - Passenger Class
  - Gender
  - Age
  - Fare
  - Port of Embarkation
  - Family Size
- Model is saved in `titanic_model.pkl`
- Cross-validation is used to prevent overfitting 