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
- Ensure device compatibility (CPU/GPU) through explicit device management.
- Set MLflow tracking URI explicitly to ensure consistent model registry access.

## Design Patterns
- Separation of concerns: data, model, training, and deployment logic are in separate files/modules.
- Config-driven: hyperparameters and settings are managed via config files or script arguments.
- Reproducibility: all experiments are tracked and can be reproduced via MLflow runs.
- Robust error handling in the Flask app for user input and model loading.
- Path management: ensuring proper module imports by modifying Python path. 
- Device agnostic: code handles both CPU and GPU environments through torch.device detection.

## Project Workflow
To run the project:
1. First, generate the dataset: `python run.py --step generate`
2. Train with default parameters: `python run.py --step train`
3. Run hyperparameter tuning: `python run.py --step tune`
4. Start the Flask app: `python run.py --step app`
5. Test the API: `python predict_client.py`
6. Access the web UI at http://localhost:5000