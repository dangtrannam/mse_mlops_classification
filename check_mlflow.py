import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List experiments
print("Available Experiments:")
for exp in client.search_experiments():
    print(f"  - {exp.experiment_id}: {exp.name}")

# Check registered models
print("\nRegistered Models:")
for rm in client.search_registered_models():
    print(f"  - {rm.name}")
    
    # Get latest versions for each model
    for mv in client.search_model_versions(f"name='{rm.name}'"):
        print(f"    * Version: {mv.version}, Stage: {mv.current_stage}, Run ID: {mv.run_id}")

# Print all tracking URIs in use
print("\nMLflow Tracking URI:", mlflow.get_tracking_uri()) 