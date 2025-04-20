import requests
import numpy as np

# Generate random features
features = np.random.randn(20).tolist()
features_str = ", ".join([str(f) for f in features])

print(f"Testing with features: {features_str}")

# Call the prediction API
url = "http://localhost:5000/predict"
response = requests.post(url, data={"features": features_str})

print(f"Response status code: {response.status_code}")
print(f"Response content: {response.text}")

# If you want to test with the error case (empty input)
print("\nTesting with empty input:")
response = requests.post(url, data={"features": ""})
print(f"Response status code: {response.status_code}")
print(f"Response content: {response.text}") 