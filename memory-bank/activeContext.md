# Active Context

## Current Work Focus
- Improving and enhancing the Flask web application.
- Ensuring robust model loading from MLflow model registry.
- Adding user-friendly features to the web interface.

## Recent Changes
- Fixed the model loading issue in the Flask application by:
  - Adding the src directory to the Python path
  - Setting the MLflow tracking URI explicitly
  - Handling device compatibility (CUDA vs CPU)
- Added a "Randomize" button to the web UI to generate random feature sets for testing.
- Implemented test scripts to diagnose and verify model loading and prediction functionality.

## Next Steps
- Improve error handling in the Flask app for better user experience.
- Add visualizations for prediction confidence.
- Consider containerizing the application for easier deployment.
- Implement automated integration tests.

## Active Decisions
- Using local MLflow tracking (`./mlruns`).
- PyTorch MLP with configurable hidden layers as the model.
- Flask for deployment with intuitive HTML UI.
- Register best model to "Staging" in MLflow Model Registry.
- Handle device compatibility issues explicitly to ensure the model works on both CPU and GPU environments. 