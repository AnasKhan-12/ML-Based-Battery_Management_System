"""
LightGBM Model for SoH (State of Health) Prediction

SoH measures battery degradation/aging (0-100%)
- 100% = Brand new battery
- 80% = End of life (typical threshold)

SoH is calculated from CYCLE-LEVEL features, not individual readings!
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
print("LIGHTGBM MODEL FOR SoH (STATE OF HEALTH) PREDICTION")
print("="*70)

# Load data
data_dir = Path(r"f:\FYP\archive (4)\cleaned_dataset")

print("\n📁 Loading data...")
charging_df = pd.read_csv(data_dir / "charging_100_cycles_with_soc.csv")
discharging_df = pd.read_csv(data_dir / "discharging_100_cycles_with_soc.csv")

print(f"  Charging rows: {len(charging_df):,}")
print(f"  Discharging rows: {len(discharging_df):,}")

# Calculate CYCLE-LEVEL features for SoH
print("\n🔧 Calculating Cycle-Level Features for SoH...")

def calculate_cycle_features(df, cycle_type='charging'):
    """
    Calculate features for each complete cycle
    These features indicate battery health/degradation
    """
    cycles = []
    
    for cycle_num in df['Cycle_Number'].unique():
        cycle_data = df[df['Cycle_Number'] == cycle_num]
        
        if len(cycle_data) < 10:  # Skip incomplete cycles
            continue
        
        # 1. Capacity (total charge in this cycle)
        if cycle_type == 'charging':
            current_col = 'Current_charge' if 'Current_charge' in cycle_data.columns else 'Current_measured'
        else:
            current_col = 'Current_load' if 'Current_load' in cycle_data.columns else 'Current_measured'
        
        time_diff = cycle_data['Time'].diff().fillna(0)
        charge_increments = cycle_data[current_col].abs() * time_diff / 3600.0
        total_capacity = charge_increments.sum()  # Ah
        
        # 2. Cycle duration
        cycle_duration = (cycle_data['Time'].max() - cycle_data['Time'].min()) / 3600.0  # hours
        
        # 3. Voltage statistics
        voltage_mean = cycle_data['Voltage_measured'].mean()
        voltage_std = cycle_data['Voltage_measured'].std()
        voltage_max = cycle_data['Voltage_measured'].max()
        voltage_min = cycle_data['Voltage_measured'].min()
        voltage_range = voltage_max - voltage_min
        
        # 4. Current statistics
        current_mean = cycle_data[current_col].abs().mean()
        current_max = cycle_data[current_col].abs().max()
        
        # 5. Temperature statistics
        temp_mean = cycle_data['Temperature_measured'].mean()
        temp_max = cycle_data['Temperature_measured'].max()
        temp_std = cycle_data['Temperature_measured'].std()
        
        # 6. Internal resistance estimate (voltage drop / current)
        # Higher resistance = degraded battery
        voltage_drop = voltage_max - voltage_min
        avg_current = current_mean
        internal_resistance = voltage_drop / avg_current if avg_current > 0 else 0
        
        # 7. Energy (Wh)
        power = cycle_data['Voltage_measured'] * cycle_data[current_col].abs()
        energy = (power * time_diff / 3600.0).sum()  # Wh
        
        cycles.append({
            'Cycle_Number': cycle_num,
            'Cycle_Type': cycle_type,
            
            # Capacity features
            'Capacity_Ah': total_capacity,
            'Energy_Wh': energy,
            'Cycle_Duration_Hours': cycle_duration,
            
            # Voltage features
            'Voltage_Mean': voltage_mean,
            'Voltage_Std': voltage_std,
            'Voltage_Max': voltage_max,
            'Voltage_Min': voltage_min,
            'Voltage_Range': voltage_range,
            
            # Current features
            'Current_Mean': current_mean,
            'Current_Max': current_max,
            
            # Temperature features
            'Temp_Mean': temp_mean,
            'Temp_Max': temp_max,
            'Temp_Std': temp_std,
            
            # Degradation indicators
            'Internal_Resistance': internal_resistance,
            'Efficiency': total_capacity / cycle_duration if cycle_duration > 0 else 0
        })
    
    return pd.DataFrame(cycles)

# Calculate features for charging and discharging cycles
print("  Processing charging cycles...")
charging_cycles = calculate_cycle_features(charging_df, 'charging')

print("  Processing discharging cycles...")
discharging_cycles = calculate_cycle_features(discharging_df, 'discharging')

# Combine
all_cycles = pd.concat([charging_cycles, discharging_cycles], ignore_index=True)
print(f"  Total cycles: {len(all_cycles)}")

# Calculate SoH labels
print("\n📊 Calculating SoH Labels...")

# SoH based on capacity fade
# Assume first cycles represent 100% health
# Later cycles show degradation

# Get reference capacity (average of first 10 cycles)
reference_capacity = all_cycles.head(10)['Capacity_Ah'].mean()
print(f"  Reference capacity (new battery): {reference_capacity:.4f} Ah")

# Calculate SoH as percentage of reference capacity
all_cycles['SoH'] = (all_cycles['Capacity_Ah'] / reference_capacity) * 100
all_cycles['SoH'] = np.clip(all_cycles['SoH'], 0, 100)

# Add cycle age (normalized)
all_cycles['Cycle_Age'] = all_cycles['Cycle_Number'] / all_cycles['Cycle_Number'].max()

# Add capacity fade
all_cycles['Capacity_Fade'] = 100 - all_cycles['SoH']

print(f"  SoH range: {all_cycles['SoH'].min():.2f}% - {all_cycles['SoH'].max():.2f}%")
print(f"  Mean SoH: {all_cycles['SoH'].mean():.2f}%")

# Feature selection for SoH model
feature_cols = [
    'Cycle_Age',
    'Capacity_Ah',
    'Energy_Wh',
    'Cycle_Duration_Hours',
    'Voltage_Mean',
    'Voltage_Range',
    'Current_Mean',
    'Temp_Mean',
    'Temp_Max',
    'Internal_Resistance',
    'Efficiency'
]

X = all_cycles[feature_cols]
y = all_cycles['SoH']

print(f"\n📊 Dataset:")
print(f"  Features: {len(feature_cols)}")
print(f"  Samples: {len(X)}")

# Train-test split
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# Train LightGBM model
print("\n🚀 Training LightGBM SoH model...")

model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

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

# Clip to valid range
y_train_pred = np.clip(y_train_pred, 0, 100)
y_test_pred = np.clip(y_test_pred, 0, 100)

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
    print(f"  {row['Feature']:25s}: {row['Importance']:.4f}")

# Save model
model_path = data_dir / "lightgbm_soh_model.pkl"
print(f"\n💾 Saving model to: {model_path}")
joblib.dump(model, model_path)

# Visualization
print("\n📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(y_test, y_test_pred, alpha=0.5, s=20)
axes[0, 0].plot([0, 100], [0, 100], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual SoH (%)', fontsize=12)
axes[0, 0].set_ylabel('Predicted SoH (%)', fontsize=12)
axes[0, 0].set_title(f'SoH: Actual vs Predicted (R² = {test_r2:.4f})', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Error distribution
errors = y_test - y_test_pred
axes[0, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Prediction Error (%)', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title(f'Error Distribution (MAE = {test_mae:.4f}%)', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature importance
top_features = importance.head(10)
axes[1, 0].barh(range(len(top_features)), top_features['Importance'].values)
axes[1, 0].set_yticks(range(len(top_features)))
axes[1, 0].set_yticklabels(top_features['Feature'].values)
axes[1, 0].set_xlabel('Importance', fontsize=12)
axes[1, 0].set_title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# 4. SoH degradation over cycles
axes[1, 1].scatter(all_cycles['Cycle_Number'], all_cycles['SoH'], alpha=0.3, s=10)
axes[1, 1].set_xlabel('Cycle Number', fontsize=12)
axes[1, 1].set_ylabel('SoH (%)', fontsize=12)
axes[1, 1].set_title('Battery Degradation Over Cycles', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = data_dir / "lightgbm_soh_results.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"  ✅ Saved: {plot_path}")

print("\n" + "="*70)
print("✅ DONE!")
print("="*70)
print(f"\n🎯 Final Test R² Score: {test_r2:.4f}")
print(f"🎯 Final Test MAE: {test_mae:.4f}%")

if test_r2 >= 0.90:
    print("\n  ✅ EXCELLENT! 90%+ accuracy achieved!")
elif test_r2 >= 0.85:
    print("\n  ✅ VERY GOOD! 85%+ accuracy")

print(f"\n💡 Note: SoH is updated WEEKLY/MONTHLY, not real-time like SoC")
print(f"   Server calculates SoH from historical cycle data")
