from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and feature names
model = joblib.load('iris_model.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the request
        data = request.get_json()
        features = [float(data[feature]) for feature in feature_names]
        
        # Make prediction
        prediction = model.predict([features])[0]
        
        # Map prediction to class name
        class_names = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = class_names[prediction]
        
        # Get prediction probabilities
        probabilities = model.predict_proba([features])[0]
        confidence = float(max(probabilities))
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                class_names[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True) 