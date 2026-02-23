"""
LightGBM Model for SoC Prediction
LightGBM is faster than XGBoost and often achieves similar accuracy
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

print("="*70)
print("LIGHTGBM MODEL FOR SoC PREDICTION")
print("="*70)

# Load data
data_dir = Path(r"f:\FYP\archive\cleaned_dataset")

print("\n📁 Loading data...")
charging_df = pd.read_csv(data_dir / "charging_100_cycles_with_soc.csv")
discharging_df = pd.read_csv(data_dir / "discharging_100_cycles_with_soc.csv")

# Add cycle type indicator
charging_df['Is_Charging'] = 1
discharging_df['Is_Charging'] = 0

# Combine
combined_df = pd.concat([charging_df, discharging_df], ignore_index=True)
print(f"  Total rows: {len(combined_df):,}")

# Feature engineering
print("\n🔧 Feature Engineering...")
combined_df['Voltage_Current_Product'] = combined_df['Voltage_measured'] * combined_df['Current_measured']
combined_df['Power'] = combined_df['Voltage_measured'] * combined_df['Current_measured']

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

# Train-test split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"  Training samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")

# Train LightGBM model
print("\n🚀 Training LightGBM model...")

model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=63,              # 2^max_depth - 1
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.log_evaluation(50)]
)

# Predictions
print("\n📈 Making predictions...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

train_r2 = r2_score(y_train, y_train_pred)
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
model_path = data_dir / "lightgbm_soc_model.pkl"
print(f"\n💾 Saving model to: {model_path}")
joblib.dump(model, model_path)

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.3, s=1)
plt.plot([0, 100], [0, 100], 'r--', lw=2)
plt.xlabel('Actual SoC (%)')
plt.ylabel('Predicted SoC (%)')
plt.title(f'LightGBM: Actual vs Predicted (R² = {test_r2:.4f})')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
errors = y_test - y_test_pred
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error (%)')
plt.ylabel('Frequency')
plt.title(f'Error Distribution (MAE = {test_mae:.4f}%)')
plt.axvline(x=0, color='r', linestyle='--', lw=2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = data_dir / "lightgbm_soc_results.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  Saved plot: {plot_path}")

print("\n" + "="*70)
print("✅ DONE!")
print("="*70)
print(f"\n🎯 Final Test R² Score: {test_r2:.4f}")
if test_r2 >= 0.90:
    print("  ✅ EXCELLENT! Achieved 90%+ accuracy target!")
