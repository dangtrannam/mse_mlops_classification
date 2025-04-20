import mlflow.pytorch
import torch
import os

# Set tracking URI explicitly to match what we saw in the check_mlflow.py output
tracking_uri = "file:///D:/Dev/MSE_projects/mse-mlops-assignment/mlruns"
mlflow.set_tracking_uri(tracking_uri)
print(f"Set tracking URI to: {tracking_uri}")

try:
    print("Attempting to load model...")
    model = mlflow.pytorch.load_model("models:/MLPClassifier/Staging")
    model.eval()
    print("Model loaded successfully!")
    
    # Test the model with random data
    print("Testing with random data...")
    random_input = torch.randn(1, 20)  # Assuming 20 features
    
    with torch.no_grad():
        output = model(random_input)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_class = torch.max(probabilities, 1)
        
    print(f"Prediction on random data: Class {predicted_class.item()}")
    print(f"Probabilities: {probabilities[0].numpy()}")
    
except Exception as e:
    print(f"Error loading or using model: {e}")
    
    # Check if model file exists in the MLflow directory
    model_path = os.path.join("mlruns", "411071367366329865", "0f3a935199ea4e5791a2c8e8c27b9155", "artifacts", "model")
    if os.path.exists(model_path):
        print(f"Model files exist at {model_path}")
        print("Files in directory:")
        for root, dirs, files in os.walk(model_path):
            for f in files:
                print(f"  - {os.path.join(root, f)}")
    else:
        print(f"Model path does not exist: {model_path}") 