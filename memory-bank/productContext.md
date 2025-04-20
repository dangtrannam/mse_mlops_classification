# Product Context

## Why This Project Exists
This project demonstrates a modern, end-to-end machine learning workflow, from data generation and model development to experiment tracking, model registry, and web deployment. It is designed for educational purposes and as a template for real-world ML projects.

## Problems It Solves
- Simplifies the process of tracking and comparing ML experiments.
- Provides a reproducible workflow for model development and deployment.
- Bridges the gap between model training and real-time inference via a web interface.
- Demonstrates handling of device compatibility issues between training and inference environments.

## How It Works
- Data is generated with scikit-learn's make_classification function.
- An MLP model is defined in PyTorch with configurable architecture.
- Training and hyperparameter tuning are performed with MLflow tracking.
- The best model is registered to the MLflow registry's Staging environment.
- A Flask web application loads the model from the registry and serves predictions.
- The web UI allows users to input feature values or generate random testing data.

## User Experience Goals
- Users can easily generate data, train models, and track results.
- The best model is always available for inference through a simple web app.
- The interface is intuitive, requiring minimal technical knowledge for predictions.
- One-click random feature generation makes testing the model simple.
- Error handling is clear and informative, guiding users on proper input formats. 