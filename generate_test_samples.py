import numpy as np

def generate_random_samples(num_samples=5):
    """Generate random feature sets for testing the prediction model"""
    for i in range(num_samples):
        # Generate random features
        features = np.random.randn(20).tolist()
        features_str = ", ".join([f"{f:.4f}" for f in features])
        
        print(f"Sample {i+1}: {features_str}")
        
if __name__ == "__main__":
    print("Here are some random feature sets you can use for testing:")
    print("=" * 80)
    generate_random_samples(5)
    print("=" * 80)
    print("Copy and paste any of these into the prediction form.") 