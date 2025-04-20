import os
import itertools
import mlflow
from train import run_training

def perform_grid_search():
    learning_rates = [0.01, 0.001, 0.0005]
    hidden_units_options = [
        [32],
        [64],
        [128],
        [64, 32]
    ]
    activations = ["relu", "tanh"]
    
    mlflow.set_experiment("mlp_hyperparameter_tuning")
    
    best_f1 = 0.0
    best_params = None
    best_run_id = None
    
    print("Starting hyperparameter grid search...")
    
    total_combinations = len(learning_rates) * len(hidden_units_options) * len(activations)
    current = 0
    
    for lr, hidden_dims, activation in itertools.product(
        learning_rates, hidden_units_options, activations
    ):
        current += 1
        print(f"\nRun {current}/{total_combinations}")
        print(f"Parameters: lr={lr}, hidden_dims={hidden_dims}, activation={activation}")
        
        with mlflow.start_run(nested=True) as run:
            _, metrics = run_training(
                learning_rate=lr,
                hidden_dims=hidden_dims,
                activation=activation,
                batch_size=32,
                num_epochs=10,
                dropout_rate=0.2,
                use_existing_run=True
            )
            
            run_id = run.info.run_id
            f1_score = metrics["f1"]
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_params = {
                    "learning_rate": lr,
                    "hidden_dims": hidden_dims,
                    "activation": activation
                }
                best_run_id = run_id
    
    print("\n" + "=" * 50)
    print("Grid search completed!")
    print(f"Best F1 Score: {best_f1:.4f}")
    print(f"Best Parameters: {best_params}")
    print(f"Best Run ID: {best_run_id}")
    
    # Register the best model in the MLflow registry
    best_model_uri = f"runs:/{best_run_id}/model"
    model_details = mlflow.register_model(
        model_uri=best_model_uri,
        name="MLPClassifier"
    )
    
    print(f"Registered best model as 'MLPClassifier' version {model_details.version}")
    
    # Transition model to staging
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name="MLPClassifier",
        version=model_details.version,
        stage="Staging"
    )
    
    print(f"Transitioned model to 'Staging' stage")
    
    return best_params, best_f1, best_run_id

if __name__ == "__main__":
    perform_grid_search() 