"""
02_model_training.py
Sri Lankan Used Vehicle Price Prediction
Algorithm: Histogram Gradient Boosting Regressor (HistGBR)

HistGBR is a modern, efficient variant of gradient boosting that uses
histogram-based split finding — similar to LightGBM — and is NOT
covered in standard ML lectures (which typically teach Decision Trees,
Random Forest, Logistic Regression, k-NN, SVM, and basic GBM).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE   = os.path.join(os.path.dirname(__file__), '..')
OUTPUT = os.path.join(BASE, 'outputs')
MODEL  = os.path.join(BASE, 'models')
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(MODEL, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD PREPROCESSED DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Preprocessed Data")
print("=" * 60)

df = pd.read_csv(os.path.join(OUTPUT, 'preprocessed_data.csv'))
print(f"Shape: {df.shape}")

# ─────────────────────────────────────────────
# 2. DEFINE FEATURES & TARGET
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Defining Feature Matrix and Target")
print("=" * 60)

# Drop non-feature columns
drop_cols = ['price', 'log_price', 'model', 'color',
             'age_group', 'fuel_efficiency', 'tax_era',
             'is_popular_model']

# Keep all numeric / encoded columns as features
feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Number of features: {len(feature_cols)}")
print("Features:", feature_cols)

X = df[feature_cols].copy()
y = df['log_price'].copy()          # predict log(price) → back-transform later

# Convert boolean columns to int
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)

# Fill any residual NaNs with column median (HistGBR handles NaN natively,
# but explicit fill is good practice for downstream tools)
X = X.fillna(X.median(numeric_only=True))

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ─────────────────────────────────────────────
# 3. TRAIN / VALIDATION / TEST SPLIT
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Train / Validation / Test Split (70 / 15 / 15)")
print("=" * 60)

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1765, random_state=42)
# 0.1765 of 85% ≈ 15% of total

print(f"Train  : {len(X_train):>5} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"Val    : {len(X_val):>5} samples ({100*len(X_val)/len(X):.1f}%)")
print(f"Test   : {len(X_test):>5} samples ({100*len(X_test)/len(X):.1f}%)")

# ─────────────────────────────────────────────
# 4. MODEL DEFINITION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Defining Histogram Gradient Boosting Regressor")
print("=" * 60)

model = HistGradientBoostingRegressor(
    max_iter=500,          # number of boosting rounds (trees)
    learning_rate=0.05,    # shrinkage rate — lower = more robust
    max_depth=6,           # depth of each tree
    min_samples_leaf=20,   # regularisation: min samples per leaf
    l2_regularization=0.1, # L2 penalty on leaf values
    max_bins=255,          # histogram bins per feature (max efficiency)
    early_stopping=True,   # stop when validation score plateaus
    validation_fraction=0.1,
    n_iter_no_change=30,
    random_state=42,
    verbose=1
)

print("\nModel Hyperparameters:")
for k, v in model.get_params().items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────────────
# 5. TRAIN MODEL
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Training the Model")
print("=" * 60)

model.fit(X_train, y_train)
print(f"\nTraining complete. Iterations used: {model.n_iter_}")

# ─────────────────────────────────────────────
# 6. EVALUATE ON ALL SPLITS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Evaluation Metrics")
print("=" * 60)

def evaluate(model, X, y, split_name):
    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y)

    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred) * 100

    print(f"\n{split_name}:")
    print(f"  R²    : {r2:.4f}  (1.00 = perfect)")
    print(f"  RMSE  : {rmse:>12,.0f} LKR")
    print(f"  MAE   : {mae:>12,.0f} LKR")
    print(f"  MAPE  : {mape:.2f}%")
    return {'split': split_name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape,
            'y_true': y_true, 'y_pred': y_pred}

res_train = evaluate(model, X_train, y_train, "TRAIN")
res_val   = evaluate(model, X_val,   y_val,   "VALIDATION")
res_test  = evaluate(model, X_test,  y_test,  "TEST")

# ─────────────────────────────────────────────
# 7. CROSS-VALIDATION (on train+val)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: 5-Fold Cross-Validation (Train+Val set)")
print("=" * 60)

cv_model = HistGradientBoostingRegressor(
    max_iter=300, learning_rate=0.05, max_depth=6,
    min_samples_leaf=20, l2_regularization=0.1,
    random_state=42
)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2   = cross_val_score(cv_model, X_train_val, y_train_val, cv=kf, scoring='r2')
cv_rmse = cross_val_score(cv_model, X_train_val, y_train_val, cv=kf,
                          scoring='neg_root_mean_squared_error')

print(f"CV R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
print(f"CV RMSE : {-cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} (log scale)")

# ─────────────────────────────────────────────
# 8. SAVE METRICS TABLE
# ─────────────────────────────────────────────
metrics_df = pd.DataFrame([
    {'Split': 'Train',      'R²': res_train['R2'], 'RMSE (LKR)': res_train['RMSE'],
     'MAE (LKR)': res_train['MAE'], 'MAPE (%)': res_train['MAPE']},
    {'Split': 'Validation', 'R²': res_val['R2'],   'RMSE (LKR)': res_val['RMSE'],
     'MAE (LKR)': res_val['MAE'],   'MAPE (%)': res_val['MAPE']},
    {'Split': 'Test',       'R²': res_test['R2'],  'RMSE (LKR)': res_test['RMSE'],
     'MAE (LKR)': res_test['MAE'],  'MAPE (%)': res_test['MAPE']},
    {'Split': 'CV (mean)',  'R²': cv_r2.mean(),    'RMSE (LKR)': '-',
     'MAE (LKR)': '-',              'MAPE (%)': '-'},
])
metrics_df.to_csv(os.path.join(OUTPUT, 'metrics_table.csv'), index=False)
print("\nMetrics table saved.")

# ─────────────────────────────────────────────
# 9. PLOTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Generating Model Performance Plots")
print("=" * 60)

plt.rcParams.update({'font.size': 11})

# --- Plot A: Actual vs Predicted (Test) ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

y_true_t = res_test['y_true']
y_pred_t = res_test['y_pred']

axes[0].scatter(y_true_t/1e6, y_pred_t/1e6, alpha=0.4, s=18,
                color='steelblue', edgecolors='none')
lim = max(y_true_t.max(), y_pred_t.max()) / 1e6
axes[0].plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='Perfect Prediction')
axes[0].set_title('Actual vs Predicted Price (Test Set)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Actual Price (Million LKR)')
axes[0].set_ylabel('Predicted Price (Million LKR)')
axes[0].legend()

# --- Plot B: Residuals ---
residuals = y_true_t - y_pred_t
axes[1].scatter(y_pred_t/1e6, residuals/1e6, alpha=0.4, s=18,
                color='coral', edgecolors='none')
axes[1].axhline(0, color='black', linewidth=1.5, linestyle='--')
axes[1].set_title('Residual Plot (Test Set)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted Price (Million LKR)')
axes[1].set_ylabel('Residual (Million LKR)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'fig08_actual_vs_predicted.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig08_actual_vs_predicted.png")

# --- Plot C: Residual Distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(residuals/1e6, bins=50, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax.set_title('Distribution of Residuals (Test Set)', fontsize=13, fontweight='bold')
ax.set_xlabel('Residual (Million LKR)')
ax.set_ylabel('Frequency')
ax.text(0.98, 0.95, f'Mean={residuals.mean()/1e6:.3f}M\nStd={residuals.std()/1e6:.3f}M',
        transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'fig09_residual_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig09_residual_distribution.png")

# --- Plot D: Metrics Bar Chart ---
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
splits = ['Train', 'Validation', 'Test']
r2_vals    = [res_train['R2'], res_val['R2'], res_test['R2']]
rmse_vals  = [res_train['RMSE']/1e6, res_val['RMSE']/1e6, res_test['RMSE']/1e6]
mape_vals  = [res_train['MAPE'], res_val['MAPE'], res_test['MAPE']]

colors = ['#2ecc71', '#3498db', '#e74c3c']

axes[0].bar(splits, r2_vals, color=colors, edgecolor='white')
axes[0].set_title('R² Score', fontweight='bold')
axes[0].set_ylim(0, 1.05)
axes[0].axhline(1, color='grey', linestyle='--', linewidth=0.8)
for i, v in enumerate(r2_vals):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')

axes[1].bar(splits, rmse_vals, color=colors, edgecolor='white')
axes[1].set_title('RMSE (Million LKR)', fontweight='bold')
for i, v in enumerate(rmse_vals):
    axes[1].text(i, v + 0.005, f'{v:.3f}', ha='center', fontweight='bold')

axes[2].bar(splits, mape_vals, color=colors, edgecolor='white')
axes[2].set_title('MAPE (%)', fontweight='bold')
for i, v in enumerate(mape_vals):
    axes[2].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontweight='bold')

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Model Performance Across Splits', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'fig10_metrics_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig10_metrics_comparison.png")

# ─────────────────────────────────────────────
# 10. SAVE MODEL & FEATURE LIST
# ─────────────────────────────────────────────
joblib.dump(model, os.path.join(MODEL, 'histgbr_model.pkl'))
joblib.dump(feature_cols, os.path.join(MODEL, 'feature_cols.pkl'))
pd.DataFrame({'feature': feature_cols}).to_csv(
    os.path.join(OUTPUT, 'feature_list.csv'), index=False)

print("\n✓ Model saved: models/histgbr_model.pkl")
print("✓ Feature list saved: outputs/feature_list.csv")
print("\n✓ Model training complete!")
