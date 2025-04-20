## 📌 Project Requirements: MLflow Project with Model Registry and Flask App

### 🧠 Objective:
Build a complete ML pipeline using **MLflow** to manage experiments, model tuning, and deployment via a simple **Flask web application**.

---

### 📊 Dataset:
- Use a **synthetic classification dataset** generated with `make_classification` from `sklearn.datasets`.

---

### 🔧 Model Development:
- Create a **basic classification model** (e.g., logistic regression, decision tree, etc.).
- Use **MLflow** to:
  - Track experiments.
  - Perform **hyperparameter tuning** over a few trials.
  - Compare and analyze model performance.
- Select the **best-performing model** based on evaluation metrics (e.g., accuracy, F1-score).
- **Register** this best model to the **MLflow Model Registry**.

---

### 🌐 Web App (Flask):
- Build a simple **Flask web application** that:
  - Loads the **best model** from the MLflow model registry.
  - Uses this model to **predict** the class of user-provided input data.
  - Always serves predictions from the **best registered model**.
