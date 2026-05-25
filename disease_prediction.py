# Disease Prediction from Medical Data - CodeAlpha ML Internship
# Heart disease prediction using classification algorithms
# Dataset: UCI Heart Disease (Cleveland)
# Author: Samah AZIZ

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("TASK 4: DISEASE PREDICTION FROM MEDICAL DATA")
print("Heart Disease Prediction")
print("=" * 60)

# --- Loading dataset ---
print("\n--- Loading Heart Disease Dataset ---")

# Load UCI Heart Disease dataset from URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

try:
    data = pd.read_csv(url, names=columns, na_values='?')
    print("[OK] Dataset loaded from UCI Repository")
except Exception:
    # Fallback: generate realistic synthetic data
    print("[WARN] UCI download failed. Generating synthetic heart disease dataset...")
    np.random.seed(42)
    n = 303

    data = pd.DataFrame({
        'age': np.random.randint(29, 77, n),
        'sex': np.random.choice([0, 1], n, p=[0.32, 0.68]),
        'cp': np.random.choice([0, 1, 2, 3], n),
        'trestbps': np.random.normal(131, 17, n).clip(94, 200).astype(int),
        'chol': np.random.normal(246, 52, n).clip(126, 564).astype(int),
        'fbs': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'restecg': np.random.choice([0, 1, 2], n, p=[0.5, 0.45, 0.05]),
        'thalach': np.random.normal(149, 23, n).clip(71, 202).astype(int),
        'exang': np.random.choice([0, 1], n, p=[0.67, 0.33]),
        'oldpeak': np.random.exponential(1.0, n).clip(0, 6.2).round(1),
        'slope': np.random.choice([0, 1, 2], n),
        'ca': np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.2, 0.12, 0.08]),
        'thal': np.random.choice([3, 6, 7], n, p=[0.55, 0.06, 0.39]),
    })

    # Generate target based on medical logic
    risk = (
        0.02 * data['age']
        - 0.01 * data['thalach']
        + 0.5 * data['exang']
        + 0.3 * data['oldpeak']
        + 0.4 * data['ca']
        + 0.3 * (data['cp'] == 0).astype(int)
        + np.random.normal(0, 0.5, n)
    )
    data['target'] = (risk > risk.median()).astype(int)
    print("[OK] Synthetic dataset generated")

# Clean data
data = data.dropna()

# Convert target: 0 = no disease, 1 = disease (original has 0-4)
data['target'] = (data['target'] > 0).astype(int)

print(f"\nDataset shape: {data.shape}")
print(f"Class distribution:\n{data['target'].value_counts()}")
print(f"\nDataset info:")
print(data.describe().round(2))

# --- EDA ---
print("\n--- Exploratory Data Analysis ---")

# Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

corr = data.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=axes[0], square=True, linewidths=0.5,
            annot_kws={'size': 7})
axes[0].set_title('Correlation Heatmap', fontsize=14, fontweight='bold')

# Target distribution
colors = ['#2ecc71', '#e74c3c']
data['target'].value_counts().plot(kind='bar', ax=axes[1], color=colors)
axes[1].set_title('Heart Disease Distribution', fontsize=14, fontweight='bold')
axes[1].set_xticklabels(['No Disease (0)', 'Disease (1)'], rotation=0)
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] EDA saved to 'eda_correlation.png'")

# Key features distribution
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
features_plot = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

for idx, feat in enumerate(features_plot):
    ax = axes[idx // 3, idx % 3]
    for label, color, name in zip([0, 1], colors, ['Healthy', 'Disease']):
        subset = data[data['target'] == label]
        ax.hist(subset[feat], bins=25, alpha=0.6, color=color, label=name)
    ax.set_title(f'{feat}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Feature Distribution by Heart Disease Status',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Feature distributions saved to 'feature_distributions.png'")

# Age vs Max Heart Rate scatter
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(data['age'], data['thalach'], c=data['target'],
                     cmap='RdYlGn_r', alpha=0.7, edgecolors='gray', s=60)
ax.set_xlabel('Age', fontsize=12)
ax.set_ylabel('Max Heart Rate (thalach)', fontsize=12)
ax.set_title('Age vs Max Heart Rate by Disease Status', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Disease (1) / Healthy (0)')
plt.tight_layout()
plt.savefig('age_vs_heartrate.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Scatter plot saved to 'age_vs_heartrate.png'")

# --- Preprocessing ---
print("\n--- Preprocessing ---")

X = data.drop('target', axis=1)
y = data['target']

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"[OK] Train set: {X_train_scaled.shape[0]} samples")
print(f"[OK] Test set:  {X_test_scaled.shape[0]} samples")
print(f"[OK] Features:  {X_train_scaled.shape[1]}")

# --- Training models ---
print("\n--- Training & Evaluation ---")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42),
    'KNN (K=5)': KNeighborsClassifier(n_neighbors=5),
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')

    results[name] = {
        'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1-Score': f1, 'ROC-AUC': auc, 'CV Mean': cv_scores.mean(),
        'y_pred': y_pred, 'y_proba': y_proba
    }

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# --- Comparison ---
print("\n--- Model Comparison ---")

comparison_df = pd.DataFrame({
    name: {k: v for k, v in vals.items() if k not in ['y_pred', 'y_proba']}
    for name, vals in results.items()
}).T.round(4)

print(comparison_df.to_string())

best_model_name = comparison_df['ROC-AUC'].idxmax()
print(f"\nBest Model: {best_model_name} "
      f"(ROC-AUC: {comparison_df.loc[best_model_name, 'ROC-AUC']:.4f})")

# --- Plots ---
print("\n--- Generating plots ---")

# --- ROC Curves ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors_list = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

for (name, vals), color in zip(results.items(), colors_list):
    fpr, tpr, _ = roc_curve(y_test, vals['y_proba'])
    axes[0].plot(fpr, tpr, linewidth=2, color=color,
                 label=f"{name} (AUC={vals['ROC-AUC']:.3f})")

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[0].set_xlabel('False Positive Rate', fontsize=12)
axes[0].set_ylabel('True Positive Rate', fontsize=12)
axes[0].set_title('ROC Curves - Heart Disease Prediction', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# --- Bar chart comparison ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(metrics))
width = 0.15

for i, (name, vals) in enumerate(results.items()):
    values = [vals[m] for m in metrics]
    axes[1].bar(x + i * width, values, width, label=name,
                color=colors_list[i], alpha=0.85)

axes[1].set_xlabel('Metrics', fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[1].set_xticks(x + width * 2)
axes[1].set_xticklabels(metrics, fontsize=10)
axes[1].legend(fontsize=8)
axes[1].set_ylim(0.4, 1.05)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Model comparison saved to 'model_comparison.png'")

# --- Confusion Matrix for best model ---
y_pred_best = results[best_model_name]['y_pred']
cm = confusion_matrix(y_test, y_pred_best)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
            xticklabels=['Healthy', 'Disease'],
            yticklabels=['Healthy', 'Disease'])
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Confusion matrix saved to 'confusion_matrix.png'")

# --- Feature Importance (Random Forest) ---
rf_model = models['Random Forest']
feature_imp = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
colors_feat = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(feature_imp)))
feature_imp.plot(kind='barh', color=colors_feat, ax=ax)
ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("[OK] Feature importance saved to 'feature_importance.png'")

# --- Final report ---
print(f"\n--- Classification Report ({best_model_name}) ---")
print(classification_report(y_test, y_pred_best, target_names=['Healthy', 'Heart Disease']))

# --- Feature interpretation ---
print("\n--- Medical Feature Interpretation ---")

feature_descriptions = {
    'age': 'Patient age (years)',
    'sex': 'Sex (1=Male, 0=Female)',
    'cp': 'Chest pain type (0-3)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl',
    'restecg': 'Resting ECG results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment',
    'ca': 'Number of major vessels colored by fluoroscopy',
    'thal': 'Thalassemia type (3=normal, 6=fixed defect, 7=reversible defect)',
}

print("\nTop 5 most important features for prediction:")
for i, (feat, imp) in enumerate(feature_imp.iloc[-5:].items()):
    desc = feature_descriptions.get(feat, feat)
    print(f"  {5-i}. {feat} ({imp:.4f}) - {desc}")

print("\nDone!")
