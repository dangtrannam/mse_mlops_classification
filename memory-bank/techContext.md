# Tech Context

## Technologies Used
- Python 3.x
- PyTorch (model implementation and training)
- scikit-learn (data generation)
- MLflow (experiment tracking, model registry)
- Flask (web deployment)
- Gradio (optional, for rapid UI prototyping)
- NumPy, tqdm, PyYAML (utilities)

## Development Setup
- Local MLflow tracking server (`./mlruns`)
- Modular directory structure: `src/`, `data/`, `app/`, `memory-bank/`
- All dependencies managed via `requirements.txt`

## Technical Constraints
- Model must be a PyTorch MLP with 1-2 hidden layers
- Data generated with `make_classification` (1000 samples, 20 features, 2 classes)
- Flask app must always serve the best model from the MLflow registry

## Dependencies
- torch
- scikit-learn
- mlflow
- flask
- numpy
- pyyaml
- tqdm
- gradio 