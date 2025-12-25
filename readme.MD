# Datacenter Energy Forecasting Analysis

### 📊 Interactive Machine Learning Notebook

This notebook provides an end-to-end framework for analyzing and predicting datacenter power consumption. It transforms raw sensor data into actionable insights and deployment-ready machine learning models.

## 🚀 How to Use This Notebook

1. **Environment Setup**: Ensure you have the required libraries installed (`pandas`, `seaborn`, `sklearn`, `joblib`).
2. **Data Placement**: Place your `synthetic_datacenter_energy_data.csv` in the same directory as this `.ipynb` file.
3. **Run All Cells**: The notebook is designed to be executed sequentially. It will handle data cleaning, generate plots, and train models in order.
4. **Artifact Generation**: At the end of the notebook, look for the `.pkl` files created in your file sidebar.

---

## 🔍 Key Analysis Phases

### 1. Data Cleaning & EDA

The notebook automatically identifies missing values and generates **10 high-resolution visualizations**. These plots allow you to see exactly how external temperature and CPU utilization drive power costs.

### 2. Model Benchmarking

We compare three distinct approaches to find the most accurate forecaster:

- **Linear Regression**: Used as a baseline for linear relationships.
- **Random Forest**: Excellent for capturing complex, non-linear interactions.
- **Gradient Boosting**: Focused on minimizing the toughest prediction errors.

### 3. Model Persistence

The final cells export the **Best Model** so it can be used in a real-time dashboard or monitoring tool without having to retrain the notebook.

---

## 📦 Required Dependencies

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

```

---

## 💡 Quick Summary of Results

Once you run the notebook, check the **Model Performance Comparison** table.

- **High scores** (close to 1.0) indicate a very reliable model.
- **Feature Importance** will tell you which sensor is your "biggest lever" for energy savings.
- **Note on Data Source** The analysis and models within this notebook are built using synthetic data. While the features (CPU utilization, cooling power, temperature) are designed to mimic real-world datacenter dynamics, the specific values, correlations, and performance metrics should be viewed as a "Proof of Concept."
