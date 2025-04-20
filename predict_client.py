import requests
import numpy as np
import argparse
import os

def generate_random_features(n_features=20):
    return np.random.randn(n_features)

def send_prediction_request(features, url="http://localhost:5000/predict"):
    features_str = ','.join(map(str, features))
    
    response = requests.post(
        url, 
        data={"features": features_str},
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    
    if response.status_code == 200:
        print("Request successful!")
        print(f"Response HTML contains prediction information.")
        print("For UI interaction, please open http://localhost:5000 in your browser.")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

def generate_data_if_needed():
    """Generate data if it doesn't exist"""
    from src.data_utils import generate_classification_data
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    data_path = os.path.join(data_dir, "classification_data.npz")
    
    if not os.path.exists(data_path):
        print("Data file not found. Generating synthetic data...")
        os.makedirs(data_dir, exist_ok=True)
        generate_classification_data(save_path=data_path)
        print("Data generation complete!")
    else:
        print(f"Data file found at {data_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the MLP Classifier Flask API")
    parser.add_argument("--random", action="store_true", help="Use random features instead of specifying them")
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature values")
    parser.add_argument("--url", type=str, default="http://localhost:5000/predict", help="API endpoint URL")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic data if needed")
    
    args = parser.parse_args()
    
    if args.generate:
        generate_data_if_needed()
    
    if args.random:
        features = generate_random_features()
        print(f"Generated random features: {features}")
    elif args.features:
        features = [float(x.strip()) for x in args.features.split(',')]
        print(f"Using provided features: {features}")
    else:
        # Default features
        features = [0.5, 1.2, -0.3, 0.8, 1.5, -0.7, 0.2, 0.9, -1.1, 0.4, 0.6, -0.5, 1.0, -0.2, 0.3, 0.7, -0.9, 1.3, -0.4, 0.1]
        print(f"Using default features: {features}")
    
    if len(features) != 20:
        print(f"Warning: Expected 20 features, got {len(features)}. This may cause an error.")
    
    send_prediction_request(features, args.url) 