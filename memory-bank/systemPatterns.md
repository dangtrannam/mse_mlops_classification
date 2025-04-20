# System Patterns

## System Architecture
- Modular pipeline: data generation, model definition, training/tuning, evaluation, registry, and deployment are separated into distinct modules.
- MLflow is used for experiment tracking, hyperparameter tuning, and model registry.
- Flask serves as the deployment layer, loading the best model from the registry for inference.

## Key Technical Decisions
- Use PyTorch for model implementation to leverage GPU acceleration and flexibility.
- Use sklearn's make_classification for reproducible, configurable synthetic data.
- Use MLflow's Python API for seamless experiment logging and model management.
- Register only the best model (by F1-score) to the registry and serve it in production.

## Design Patterns
- Separation of concerns: data, model, training, and deployment logic are in separate files/modules.
- Config-driven: hyperparameters and settings are managed via config files or script arguments.
- Reproducibility: all experiments are tracked and can be reproduced via MLflow runs.
- Robust error handling in the Flask app for user input and model loading. 

To run the project:
First, generate the dataset: mlflow run . -e generate_data
Run hyperparameter tuning: mlflow run . -e tune
Start the Flask app: mlflow run . -e app
Test the API: python predict_client.py
Access the web UI at http://localhost:5000