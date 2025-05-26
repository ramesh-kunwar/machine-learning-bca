# Iris Flower Classifier

This is a machine learning web application that predicts iris flower species based on their measurements. The model achieves over 90% accuracy while preventing overfitting through careful model selection and hyperparameter tuning.

## Features

- Random Forest Classifier with 90%+ accuracy
- Cross-validation to prevent overfitting
- Modern web interface
- Real-time predictions with confidence scores
- RESTful API endpoint

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train_model.py
```

4. Run the web application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Enter the measurements of an iris flower:
   - Sepal length (cm)
   - Sepal width (cm)
   - Petal length (cm)
   - Petal width (cm)

2. Click "Predict Species" to get the prediction

3. View the predicted species and confidence scores for each possible class

## Model Details

- Algorithm: Random Forest Classifier
- Features: Sepal length, Sepal width, Petal length, Petal width
- Target: Iris species (Setosa, Versicolor, Virginica)
- Model saved as: `iris_model.pkl`
- Feature names saved as: `feature_names.pkl`

## API Endpoint

The model can be accessed via a REST API endpoint:

- URL: `/predict`
- Method: POST
- Content-Type: application/json
- Request body example:
```json
{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
}
``` 