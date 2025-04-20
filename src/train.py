import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import argparse
from mlflow.models.signature import infer_signature
from contextlib import nullcontext

from model import MLP

def load_data(data_path=None, test_size=0.2, batch_size=32):
    if data_path is None:
        # Use a path relative to the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")
        data_path = os.path.join(data_dir, "classification_data.npz")
    
    data = np.load(data_path)
    X, y = data["X"], data["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_train.shape[1], len(np.unique(y))

def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    num_epochs=10,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model = model.to(device)
    
    best_f1 = 0.0
    metrics = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1": []}
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        with tqdm(train_loader, unit="batch") as t:
            t.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                t.set_postfix(loss=loss.item())
        
        avg_train_loss = total_loss / len(train_loader)
        metrics["train_loss"].append(avg_train_loss)
        
        val_metrics = evaluate_model(model, test_loader, criterion, device)
        metrics["val_loss"].append(val_metrics["loss"])
        metrics["val_accuracy"].append(val_metrics["accuracy"])
        metrics["val_f1"].append(val_metrics["f1"])
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['f1']:.4f}")
        
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_metrics["loss"], step=epoch)
        mlflow.log_metric("val_accuracy", val_metrics["accuracy"], step=epoch)
        mlflow.log_metric("val_f1", val_metrics["f1"], step=epoch)
        
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            mlflow.log_metric("best_f1", best_f1)
    
    return model, metrics

def evaluate_model(model, test_loader, criterion=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if criterion is not None:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")
    
    metrics = {
        "accuracy": accuracy,
        "f1": f1,
        "loss": total_loss / len(test_loader) if criterion is not None else 0
    }
    
    return metrics

def parse_list(s):
    try:
        return [int(item) for item in s.strip('[]').split(',')]
    except:
        return [64, 32]  # Default

def run_training(
    learning_rate=0.001,
    hidden_dims=[64, 32],
    activation="relu",
    batch_size=32,
    num_epochs=10,
    dropout_rate=0.2,
    use_existing_run=False
):
    if not use_existing_run:
        mlflow.set_experiment("mlp_classification")
        run_context = mlflow.start_run()
    else:
        run_context = nullcontext()
    
    with run_context:
        if not use_existing_run:
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("hidden_dims", hidden_dims)
            mlflow.log_param("activation", activation)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("dropout_rate", dropout_rate)
        else:
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("hidden_dims", hidden_dims)
            mlflow.log_param("activation", activation)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("dropout_rate", dropout_rate)
        
        train_loader, test_loader, input_dim, output_dim = load_data(batch_size=batch_size)
        
        model = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            dropout_rate=dropout_rate
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {device}")
        
        trained_model, _ = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            device=device
        )
        
        final_metrics = evaluate_model(trained_model, test_loader, criterion, device)
        print(f"Final Metrics: Accuracy = {final_metrics['accuracy']:.4f}, F1 = {final_metrics['f1']:.4f}")
        
        # Create a sample input for the model signature
        sample_input = torch.randn(1, input_dim).to(device)
        
        # Get model prediction on the sample input
        with torch.no_grad():
            sample_output = trained_model(sample_input)
            
        # Convert tensors to numpy for MLflow
        sample_input_np = sample_input.cpu().numpy()
        sample_output_np = sample_output.cpu().numpy()
        
        # Infer the model signature
        signature = infer_signature(
            sample_input_np,
            sample_output_np
        )
        
        # Log the model with the signature and input example
        mlflow.pytorch.log_model(
            trained_model, 
            "model",
            signature=signature,
            input_example=sample_input_np
        )
        
        return trained_model, final_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLP Classifier")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--hidden_dims", type=str, default="[64,32]", help="Hidden dimensions as comma-separated list in brackets, e.g., [64,32]")
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"], help="Activation function")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate")
    
    args = parser.parse_args()
    
    hidden_dims = parse_list(args.hidden_dims)
    
    run_training(
        learning_rate=args.learning_rate,
        hidden_dims=hidden_dims,
        activation=args.activation,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        dropout_rate=args.dropout_rate
    ) 