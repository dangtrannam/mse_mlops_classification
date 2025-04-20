import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import mlflow.pytorch
import torch

app = Flask(__name__)

# Load the model from MLflow model registry
try:
    model = mlflow.pytorch.load_model("models:/MLPClassifier/Staging")
    model.eval()
    print("Successfully loaded model from MLflow model registry")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded. Please check MLflow registry."}), 500
        
        features_str = request.form.get('features', '')
        features = [float(x.strip()) for x in features_str.split(',') if x.strip()]
        
        if len(features) != 20:
            return render_template('result.html', prediction=None, 
                                  error=f"Expected 20 features, got {len(features)}. Please provide exactly 20 comma-separated values.")
        
        # Convert to tensor
        X = torch.FloatTensor([features])
        
        # Make prediction
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted_class = torch.max(probabilities, 1)
            confidence = probabilities[0][predicted_class].item() * 100
        
        class_label = "Class 1" if predicted_class.item() == 1 else "Class 0"
        
        return render_template('result.html', 
                              prediction=class_label, 
                              confidence=f"{confidence:.2f}%", 
                              features=features,
                              error=None)
    
    except Exception as e:
        return render_template('result.html', prediction=None, error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 