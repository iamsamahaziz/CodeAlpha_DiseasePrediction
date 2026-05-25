# CodeAlpha - Disease Prediction from Medical Data 🏥

## Overview
This project predicts the **possibility of heart disease** based on patient medical data using classification techniques. Built as part of the **CodeAlpha Machine Learning Internship**.

## Dataset
**Heart Disease Dataset** from UCI Machine Learning Repository (Cleveland).
- 303 patient records
- 13 medical features + 1 target variable

## Features
| Feature | Description |
|---------|-------------|
| `age` | Patient age |
| `sex` | Sex (1=Male, 0=Female) |
| `cp` | Chest pain type (0-3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise induced angina |
| `oldpeak` | ST depression |
| `slope` | Slope of peak exercise ST |
| `ca` | Major vessels colored by fluoroscopy |
| `thal` | Thalassemia type |

## Algorithms Used
- Logistic Regression
- SVM (RBF Kernel)
- Random Forest
- Gradient Boosting (XGBoost-style)
- KNN

## Evaluation
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 5-Fold Stratified Cross-Validation
- Confusion Matrix & Feature Importance

## How to Run
```bash
pip install -r requirements.txt
python disease_prediction.py
```

## Generated Outputs
- `eda_correlation.png` — Correlation heatmap
- `feature_distributions.png` — Feature distributions by disease status
- `age_vs_heartrate.png` — Scatter plot
- `model_comparison.png` — ROC curves and bar chart comparison
- `confusion_matrix.png` — Best model confusion matrix
- `feature_importance.png` — Random Forest feature importance

## Author
**Samah AZIZ** — CodeAlpha ML Internship
