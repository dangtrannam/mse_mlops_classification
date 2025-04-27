# MLOps Pipeline with PyTorch, MLflow and Flask

This project demonstrates a complete MLOps pipeline that includes:
- Data generation
- PyTorch model implementation (MLP classifier)
- Training and evaluation
- Hyperparameter tuning
- Experiment tracking with MLflow
- Model registry
- Flask web application for serving predictions

## Project Structure

```
.
├── app/                      # Flask web application
│   ├── app.py                # Main Flask application
│   └── templates/            # HTML templates
│       ├── index.html        # Input form
│       └── result.html       # Prediction results
├── data/                     # Data directory (created automatically)
├── src/                      # Source code
│   ├── data_utils.py         # Data generation utilities
│   ├── model.py              # PyTorch MLP model
│   ├── train.py              # Training script
│   └── tune.py               # Hyperparameter tuning
├── memory-bank/              # Documentation
├── conda.yaml                # Conda environment specification
├── python_env.yaml           # Python environment for MLflow
├── MLproject                 # MLflow project definition
├── predict_client.py         # Test client for Flask API
├── requirements.txt          # Python dependencies
└── run.py                    # Script to run the pipeline
```

## Setup

1. Install dependencies:

```bash
# Option 1: Using pip
pip install -r requirements.txt

# Option 2: Using conda
conda env create -f conda.yaml
conda activate mlp-classifier
```

## Running the Pipeline

### Option 1: Run each step individually

You can run each step of the pipeline manually:

```bash
# Generate data
python run.py --step generate

# Train model with default parameters
python run.py --step train

# Perform hyperparameter tuning
python run.py --step tune

# Run the Flask app
python run.py --step app
```

### Option 2: Run all steps at once

```bash
python run.py --step all
```

### Option 3: Run using MLflow

```bash
# Generate data
mlflow run . -e generate_data

# Train model with default parameters
mlflow run . -e train

# Tune hyperparameters
mlflow run . -e tune

# Run the Flask app
mlflow run . -e app
```

## Using the Web Application

1. Start the Flask application:
   ```
   python run.py --step app
   ```

2. Open your web browser and navigate to http://localhost:5000

3. Enter 20 comma-separated feature values in the form and click "Predict"

4. View the prediction result

## Testing the API

You can test the API directly using the provided client:

```bash
# Use default features
python predict_client.py

# Use random features
python predict_client.py --random

# Specify custom features (20 comma-separated values)
python predict_client.py --features "0.5,1.2,-0.3,0.8,1.5,-0.7,0.2,0.9,-1.1,0.4,0.6,-0.5,1.0,-0.2,0.3,0.7,-0.9,1.3,-0.4,0.1"

# Generate data if it doesn't exist yet
python predict_client.py --generate
```

## Viewing Experiment Results

To view the MLflow tracking UI:

```bash
mlflow ui --port 5001
```

Then open your browser and navigate to http://localhost:5001 to explore your experiments. 