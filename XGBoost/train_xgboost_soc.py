"""
XGBoost Model for SoC Prediction
XGBoost is excellent for battery prediction - typically achieves 90%+ R² score
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("XGBOOST MODEL FOR SoC PREDICTION")
print("="*70)

# Load data
data_dir = Path(r"f:\FYP\archive (4)\cleaned_dataset")

print("\n📁 Loading data...")
charging_df = pd.read_csv(data_dir / "charging_100_cycles_with_soc.csv")
discharging_df = pd.read_csv(data_dir / "discharging_100_cycles_with_soc.csv")

print(f"  Charging rows: {len(charging_df):,}")
print(f"  Discharging rows: {len(discharging_df):,}")

# Combine both datasets
combined_df = pd.concat([charging_df, discharging_df], ignore_index=True)
print(f"  Total rows: {len(combined_df):,}")

# Feature engineering
print("\n🔧 Feature Engineering...")

# Add cycle type indicator
charging_df['Is_Charging'] = 1
discharging_df['Is_Charging'] = 0
combined_df = pd.concat([charging_df, discharging_df], ignore_index=True)

# Additional features
combined_df['Voltage_Current_Product'] = combined_df['Voltage_measured'] * combined_df['Current_measured']
combined_df['Power'] = combined_df['Voltage_measured'] * combined_df['Current_measured']

# Select features
feature_cols = [
    'Voltage_measured',
    'Current_measured', 
    'Temperature_measured',
    'Is_Charging',
    'Voltage_Current_Product',
    'Power'
]

X = combined_df[feature_cols]
y = combined_df['SoC']

print(f"  Features: {feature_cols}")
print(f"  Feature shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# Train-test split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")

# Train XGBoost model
print("\n🚀 Training XGBoost model...")
print("  This may take a few minutes...")

model = xgb.XGBRegressor(
    n_estimators=500,           # More trees = better accuracy
    max_depth=8,                # Deep enough to capture patterns
    learning_rate=0.05,         # Slower learning = better generalization
    subsample=0.8,              # Use 80% of data per tree
    colsample_bytree=0.8,       # Use 80% of features per tree
    min_child_weight=3,         # Prevent overfitting
    gamma=0.1,                  # Regularization
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,             # L2 regularization
    random_state=42,
    n_jobs=-1,                  # Use all CPU cores
    tree_method='hist',         # Faster training
    eval_metric='rmse'
)

# Train with early stopping
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    verbose=50  # Print progress every 50 iterations
)

# Predictions
print("\n📈 Making predictions...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Training metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print("\n📊 Training Set Performance:")
print(f"  MAE:  {train_mae:.4f}%")
print(f"  RMSE: {train_rmse:.4f}%")
print(f"  R²:   {train_r2:.4f}")

# Test metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n📊 Test Set Performance:")
print(f"  MAE:  {test_mae:.4f}%")
print(f"  RMSE: {test_rmse:.4f}%")
print(f"  R²:   {test_r2:.4f}")

# Feature importance
print("\n🔍 Feature Importance:")
importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in importance.iterrows():
    print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")

# Save model
model_path = data_dir / "xgboost_soc_model.json"
print(f"\n💾 Saving model to: {model_path}")
model.save_model(str(model_path))

# Also save as pickle for easier loading
pickle_path = data_dir / "xgboost_soc_model.pkl"
joblib.dump(model, pickle_path)
print(f"💾 Saved pickle to: {pickle_path}")

# Visualizations
print("\n📊 Creating visualizations...")

# 1. Actual vs Predicted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.3, s=1)
plt.plot([0, 100], [0, 100], 'r--', lw=2)
plt.xlabel('Actual SoC (%)')
plt.ylabel('Predicted SoC (%)')
plt.title(f'XGBoost: Actual vs Predicted (R² = {test_r2:.4f})')
plt.grid(True, alpha=0.3)

# 2. Error distribution
plt.subplot(1, 2, 2)
errors = y_test - y_test_pred
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error (%)')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (MAE = {test_mae:.4f}%)')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = data_dir / "xgboost_soc_results.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  Saved plot: {plot_path}")

# 3. Feature importance plot
plt.figure(figsize=(10, 6))
importance.plot(x='Feature', y='Importance', kind='barh', legend=False)
plt.xlabel('Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
importance_plot_path = data_dir / "xgboost_feature_importance.png"
plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
print(f"  Saved plot: {importance_plot_path}")

print("\n" + "="*70)
print("✅ DONE!")
print("="*70)
print(f"\n🎯 Final Test R² Score: {test_r2:.4f}")
if test_r2 >= 0.90:
    print("  ✅ EXCELLENT! Achieved 90%+ accuracy target!")
elif test_r2 >= 0.85:
    print("  ✅ VERY GOOD! Close to 90% target")
else:
    print("  ⚠️  Below target - may need hyperparameter tuning")

print(f"\n📁 Model saved to: {model_path}")
print(f"📁 Visualizations saved to: {data_dir}")
