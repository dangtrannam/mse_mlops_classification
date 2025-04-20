# Tech Context

## Technologies Used
- Python 3.x
- PyTorch (model implementation and training)
- scikit-learn (data generation)
- MLflow (experiment tracking, model registry)
- Flask (web deployment)
- NumPy, tqdm, PyYAML (utilities)
- JavaScript (for UI randomization functionality)

## Development Setup
- Local MLflow tracking server (`./mlruns`)
- Modular directory structure: `src/`, `data/`, `app/`, `memory-bank/`
- All dependencies managed via `requirements.txt`
- Python path modification for proper module imports
- Device detection (CPU/GPU) for compatibility

## Technical Constraints
- Model must be a PyTorch MLP with configurable hidden layers
- Data generated with `make_classification` (1000 samples, 20 features, 2 classes)
- Flask app must always serve the best model from the MLflow registry
- Model must be compatible with both CPU and GPU inference

## Dependencies
- torch
- scikit-learn
- mlflow
- flask
- numpy
- pyyaml
- tqdm
- requests (for API testing)

## Debugging Tools
- check_mlflow.py: Script to inspect MLflow experiments and models
- test_model_load.py: Script to test model loading from registry
- test_prediction.py: Script to test the prediction API endpoint
- generate_test_samples.py: Script to generate random feature sets for testing 