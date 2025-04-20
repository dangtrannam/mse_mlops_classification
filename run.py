#!/usr/bin/env python
"""
Run the MLOps pipeline step by step without relying on MLproject

Usage:
    python run.py --step [generate|train|tune|app|all]
"""

import os
import argparse
import subprocess
import sys

def ensure_data_directory():
    """Ensure the data directory exists"""
    os.makedirs("data", exist_ok=True)
    print("✅ Ensured data directory exists")
    
def generate_data():
    """Generate synthetic data for the model"""
    print("\n🚀 Generating synthetic data...")
    subprocess.run([sys.executable, "src/data_utils.py"])
    
def train_model():
    """Train the MLP model with default parameters"""
    print("\n🚀 Training the model with default parameters...")
    subprocess.run([
        sys.executable, "src/train.py", 
        "--learning_rate", "0.001",
        "--activation", "relu",
        "--num_epochs", "10"
    ])
    
def tune_model():
    """Perform hyperparameter tuning"""
    print("\n🚀 Performing hyperparameter tuning...")
    subprocess.run([sys.executable, "src/tune.py"])
    
def run_app():
    """Run the Flask application"""
    print("\n🚀 Starting the Flask application...")
    subprocess.run([sys.executable, "app/app.py"])
    
def run_all():
    """Run all steps in sequence"""
    ensure_data_directory()
    generate_data()
    train_model()
    tune_model()
    run_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MLOps pipeline step by step")
    parser.add_argument("--step", choices=["generate", "train", "tune", "app", "all"], 
                        default="all", help="Which step to run")
    
    args = parser.parse_args()
    
    # Create necessary directories
    ensure_data_directory()
    
    # Run the specified step
    if args.step == "generate":
        generate_data()
    elif args.step == "train":
        train_model()
    elif args.step == "tune":
        tune_model()
    elif args.step == "app":
        run_app()
    elif args.step == "all":
        run_all()
    else:
        parser.print_help() 