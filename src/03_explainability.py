"""
03_explainability.py
Sri Lankan Used Vehicle Price Prediction
XAI Methods: Permutation Importance, Partial Dependence Plots,
             SHAP-style manual analysis using HistGBR internals
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE   = os.path.join(os.path.dirname(__file__), '..')
OUTPUT = os.path.join(BASE, 'outputs')
MODEL  = os.path.join(BASE, 'models')

# ─────────────────────────────────────────────
# 1. LOAD MODEL & DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Model and Data")
print("=" * 60)

model        = joblib.load(os.path.join(MODEL, 'histgbr_model.pkl'))
feature_cols = joblib.load(os.path.join(MODEL, 'feature_cols.pkl'))
df           = pd.read_csv(os.path.join(OUTPUT, 'preprocessed_data.csv'))

# Rebuild X, y identically to training script
drop_cols = ['price', 'log_price', 'model', 'color',
             'age_group', 'fuel_efficiency', 'tax_era', 'is_popular_model']
X = df[feature_cols].copy()
y = df['log_price'].copy()
bool_cols = X.select_dtypes(include='bool').columns
X[bool_cols] = X[bool_cols].astype(int)
X = X.fillna(X.median(numeric_only=True))

# Same split as training (same random_state guarantees identical test set)
X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.1765, random_state=42)

print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

plt.rcParams.update({'font.size': 11})

# ─────────────────────────────────────────────
# 2. PERMUTATION FEATURE IMPORTANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Permutation Feature Importance (on Test Set)")
print("=" * 60)

print("Computing permutation importance (n_repeats=10)...")
perm = permutation_importance(model, X_test, y_test,
                              n_repeats=10, random_state=42,
                              scoring='r2', n_jobs=-1)

perm_df = pd.DataFrame({
    'Feature':    feature_cols,
    'Importance': perm.importances_mean,
    'Std':        perm.importances_std
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\nTop 20 Most Important Features:")
print(perm_df.head(20).to_string(index=False))

perm_df.to_csv(os.path.join(OUTPUT, 'permutation_importance.csv'), index=False)

# Plot top 20
top20 = perm_df.head(20).sort_values('Importance')
fig, ax = plt.subplots(figsize=(11, 8))
colors = ['#e74c3c' if imp > 0 else '#95a5a6' for imp in top20['Importance']]
bars = ax.barh(top20['Feature'], top20['Importance'], xerr=top20['Std'],
               color=colors, edgecolor='white', capsize=3, height=0.75)
ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Top 20 Features — Permutation Importance (Test Set)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Mean Decrease in R² when Feature is Permuted')
ax.set_ylabel('Feature')

# Highlight groups
highlight = {'price_per_km': '#e74c3c', 'mileage': '#e67e22',
             'vehicle_age': '#e67e22', 'year': '#3498db',
             'engine_capacity': '#2ecc71', 'features_count': '#9b59b6'}
for bar, name in zip(bars, top20['Feature']):
    if name in highlight:
        bar.set_facecolor(highlight[name])

legend_patches = [
    mpatches.Patch(color='#e74c3c', label='Price-related'),
    mpatches.Patch(color='#e67e22', label='Age/Mileage'),
    mpatches.Patch(color='#3498db', label='Vehicle Year'),
    mpatches.Patch(color='#2ecc71', label='Engine'),
    mpatches.Patch(color='#9b59b6', label='Features'),
]
ax.legend(handles=legend_patches, loc='lower right', framealpha=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'fig11_permutation_importance.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig11_permutation_importance.png")

# ─────────────────────────────────────────────
# 3. PARTIAL DEPENDENCE PLOTS (PDPs)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Partial Dependence Plots (PDPs)")
print("=" * 60)

# Select key numeric features for PDP (top features from permutation)
key_numeric_features = []
numeric_candidates = ['vehicle_age', 'mileage', 'year', 'engine_capacity',
                      'features_count', 'mileage_per_year', 'price_per_km',
                      'fuel_efficiency_enc', 'tax_era_enc']

for f in numeric_candidates:
    if f in feature_cols:
        key_numeric_features.append(f)
    if len(key_numeric_features) == 6:
        break

print(f"PDP features: {key_numeric_features}")

feature_indices = [feature_cols.index(f) for f in key_numeric_features]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()

for i, (feat, idx) in enumerate(zip(key_numeric_features, feature_indices)):
    disp = PartialDependenceDisplay.from_estimator(
        model, X_train, features=[idx],
        feature_names=feature_cols,
        ax=axes_flat[i],
        line_kw={"color": "steelblue", "linewidth": 2},
        grid_resolution=50
    )
    axes_flat[i].set_title(f'PDP: {feat.replace("_", " ").title()}',
                            fontsize=12, fontweight='bold')
    axes_flat[i].set_xlabel(feat.replace('_', ' ').title())
    axes_flat[i].set_ylabel('Partial Dependence\n(log price)')
    axes_flat[i].spines['top'].set_visible(False)
    axes_flat[i].spines['right'].set_visible(False)

plt.suptitle('Partial Dependence Plots — Effect of Key Features on Predicted Price',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'fig12_partial_dependence_plots.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig12_partial_dependence_plots.png")

# ─────────────────────────────────────────────
# 4. MANUAL SHAP-STYLE ANALYSIS
#    (Marginal contribution of top features)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: SHAP-Style Marginal Analysis")
print("=" * 60)

# Use 200-sample subset for interpretability analysis
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), size=min(200, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx].copy()
y_sample = y_test.iloc[sample_idx].copy()

baseline_pred = model.predict(X_sample)

# For each top feature, measure prediction change when feature is shifted ±1 std
top10_features = perm_df.head(10)['Feature'].tolist()
shap_approx = {}

for feat in top10_features:
    if feat not in feature_cols:
        continue
    std_val = X_sample[feat].std()
    if std_val == 0:
        continue
    X_up = X_sample.copy()
    X_up[feat] = X_up[feat] + std_val
    pred_up = model.predict(X_up)
    marginal = np.mean(pred_up - baseline_pred)
    shap_approx[feat] = marginal

shap_df = pd.DataFrame({
    'Feature': list(shap_approx.keys()),
    'Marginal Effect': list(shap_approx.values())
}).sort_values('Marginal Effect', key=abs, ascending=False)

print("Marginal Effect (+1 Std) on log(price):")
print(shap_df.to_string(index=False))

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
colors_shap = ['#e74c3c' if v > 0 else '#3498db' for v in shap_df['Marginal Effect']]
ax.barh(shap_df['Feature'][::-1], shap_df['Marginal Effect'][::-1],
        color=colors_shap[::-1], edgecolor='white', height=0.7)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Marginal Feature Effect on log(Price)\n(+1 Std Shift, 200-sample estimate)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Change in Predicted log(Price)')
pos_patch = mpatches.Patch(color='#e74c3c', label='Increases Price')
neg_patch = mpatches.Patch(color='#3498db', label='Decreases Price')
ax.legend(handles=[pos_patch, neg_patch])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'fig13_shap_style_analysis.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig13_shap_style_analysis.png")

# ─────────────────────────────────────────────
# 5. BRAND-LEVEL PRICE PREDICTION ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Brand-Level Prediction Analysis")
print("=" * 60)

df_raw = pd.read_csv(os.path.join(OUTPUT, 'cleaned_data.csv'))
y_pred_all = np.expm1(model.predict(X))
df_analysis = df_raw.copy()

# Align indices
if len(df_analysis) > len(y_pred_all):
    df_analysis = df_analysis.iloc[:len(y_pred_all)].copy()
df_analysis['predicted_price'] = y_pred_all[:len(df_analysis)]
df_analysis['actual_price']    = np.expm1(y.values[:len(df_analysis)])
df_analysis['abs_error']       = abs(df_analysis['predicted_price'] -
                                     df_analysis['actual_price'])
df_analysis['pct_error']       = (df_analysis['abs_error'] /
                                   df_analysis['actual_price']) * 100

brand_analysis = df_analysis.groupby('brand').agg(
    Count=('brand', 'count'),
    Avg_Actual=('actual_price', 'mean'),
    Avg_Predicted=('predicted_price', 'mean'),
    Avg_MAPE=('pct_error', 'mean')
).round(2).sort_values('Avg_MAPE')

print(brand_analysis.to_string())
brand_analysis.to_csv(os.path.join(OUTPUT, 'brand_analysis.csv'))

# ─────────────────────────────────────────────
# 6. FEATURE INTERACTION: VEHICLE AGE × FUEL TYPE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Feature Interaction Plot")
print("=" * 60)

# Merge prediction with raw data for fuel type grouping
df_test_raw = df_raw.iloc[X_test.index].copy() if len(df_raw) > max(X_test.index) else df_raw.iloc[:len(X_test)].copy()
y_pred_test = np.expm1(model.predict(X_test))
y_true_test = np.expm1(y_test.values)

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(y_true_test/1e6, y_pred_test/1e6, alpha=0.35, s=20,
           color='steelblue', edgecolors='none', label='All vehicles')
lim_val = max(y_true_test.max(), y_pred_test.max()) / 1e6
ax.plot([0, lim_val], [0, lim_val], 'r--', linewidth=2, label='Perfect fit')
ax.set_xlabel('Actual Price (Million LKR)', fontsize=12)
ax.set_ylabel('Predicted Price (Million LKR)', fontsize=12)
ax.set_title('Model Calibration on Test Set\n(Actual vs. Predicted — LKR)',
             fontsize=13, fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT, 'fig14_model_calibration.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig14_model_calibration.png")

print("\n✓ Explainability analysis complete!")

