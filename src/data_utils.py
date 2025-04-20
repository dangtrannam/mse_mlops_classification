import os
import numpy as np
from sklearn.datasets import make_classification


def generate_classification_data(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    n_classes=2,
    random_state=42,
    save_path=None
):
    if save_path is None:
        # Use a path relative to the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
        
        # Create the data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        save_path = os.path.join(data_dir, "classification_data.npz")
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        random_state=random_state,
    )
    np.savez(save_path, X=X, y=y)
    print(f"Data saved to {save_path}")
    return X, y


if __name__ == "__main__":
    generate_classification_data() 