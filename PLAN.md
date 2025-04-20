# Project Plan: MLflow, PyTorch, and Flask Integration

## 🧠 Objective:
Build an ML pipeline using PyTorch for modeling, MLflow for experiment tracking, hyperparameter tuning, and model registry, and Flask for deploying the best model via a web app.

## 🛠️ Proposed Approach:
Create a modular project structure. Use `sklearn` for data generation, `PyTorch` for the MLP model, `MLflow` (via its Python API) for the MLOps lifecycle (local tracking, registry), and `Flask` with HTML templates for the web interface. Utilize an `MLproject` file for reproducibility.

## 📋 Detailed Steps:

1.  [x] **Setup Project Structure & Environment:**
    - [x] Create directories: `data/`, `src/`, `app/`, `app/templates/`, `memory-bank/`.
    - [x] Initialize `requirements.txt` (`torch`, `sklearn`, `mlflow`, `flask`, `numpy`, `pyyaml`).
2.  [x] **Create Initial Memory Bank Files:**
    - [x] Generate basic `projectbrief.md`, `techContext.md`, `productContext.md`, `systemPatterns.md`, `activeContext.md`, and `progress.md` in `memory-bank/`.
3.  [x] **Implement Data Generation:**
    - [x] Create `src/data_utils.py`.
    - [x] Use `sklearn.datasets.make_classification` (params: `n_samples=1000`, `n_features=20`, `n_informative=10`, `n_redundant=5`, `n_classes=2`, `random_state=42`).
    - [x] Save data to `data/`.
4.  [x] **Define PyTorch Model:**
    - [x] Create `src/model.py`.
    - [x] Define MLP class (`nn.Module`) with 1-2 hidden layers and configurable activation (`relu`/`tanh`).
5.  [x] **Implement Base Training Script:**
    - [x] Create `src/train.py`.
    - [x] Include data loading, model instantiation, training loop, basic evaluation (accuracy, F1).
    - [x] Integrate MLflow: `start_run()`, `log_param()`, `log_metric()`, `mlflow.pytorch.log_model()`.
6.  [x] **Add Hyperparameter Tuning:**
    - [x] Enhance `src/train.py` or create `src/tune.py`.
    - [x] Loop through hyperparameter combinations (LR: `[0.01, 0.001, 0.0005]`, Hidden Units: `[32, 64, 128]`, Activation: `['relu', 'tanh']`).
    - [x] Run each combination as a separate MLflow run.
7.  [x] **Implement Model Selection & Registration:**
    - [x] Add logic to query MLflow runs (`mlflow.search_runs()`).
    - [x] Identify the best run (e.g., based on validation F1).
    - [x] Register the best model artifact (`mlflow.register_model()`) as "MLPClassifier" in the "Staging" stage.
8.  [x] **Develop Flask Application:**
    - [x] Create `app/app.py`.
    - [x] Load the latest "Staging" model (`models:/MLPClassifier/Staging`).
    - [x] Create `app/templates/index.html` (input form for comma-separated features).
    - [x] Implement `/predict` route: preprocess input, infer, display result on `app/templates/result.html`.
9.  [x] **Create `MLproject` File:**
    - [x] Create `MLproject` in the root.
    - [x] Define entry points (`generate_data`, `train`).
    - [x] Specify environment (`requirements.txt`).
10. [x] **(Optional) Create Test Script:**
    - [x] Create `predict_client.py` to send test requests to the Flask `/predict` endpoint.

## ✅ Verification Strategy:
*   Use `mlflow ui` to inspect runs and registry.
*   Run steps via `mlflow run . -e <entry_point>`.
*   Manual browser testing of the Flask app.
*   Optional: Automated testing with `predict_client.py`.

## ⚙️ Configuration Summary:
*   **MLflow Tracking:** Local `./mlruns` directory.
*   **Model:** PyTorch MLP (1-2 hidden layers).
*   **Hyperparameters:** Learning Rate, Hidden Units, Activation Function.
*   **Flask UI:** Basic HTML Templates.
*   **Model Registry:** Target "Staging" stage.
*   **Dataset:** `make_classification` (1000 samples, 20 features, 2 classes).

## 🔑 Key Files/Modules:
*   `memory-bank/*`
*   `data/`
*   `src/data_utils.py`
*   `src/model.py`
*   `src/train.py`
*   `src/tune.py`
*   `app/app.py`
*   `app/templates/index.html`
*   `app/templates/result.html`
*   `requirements.txt`
*   `MLproject`
*   `python_env.yaml`
*   `PLAN.md`
*   `predict_client.py` 