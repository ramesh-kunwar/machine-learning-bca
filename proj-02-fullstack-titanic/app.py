from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the model and preprocessing information
with open('titanic_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
features = model_data['features']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in the correct order
        input_data = np.array([[
            float(data['pclass']),
            float(data['sex']),  # 0 for female, 1 for male
            float(data['age']),
            float(data['fare']),
            float(data['embarked']),  # 0 for C, 1 for Q, 2 for S
            float(data['familySize'])
        ]])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability[1]),
            'message': 'Survived' if prediction == 1 else 'Did not survive'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=4000) 