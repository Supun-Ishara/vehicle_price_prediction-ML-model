"""
01_data_preprocessing.py
Sri Lankan Used Vehicle Price Prediction
ML Assignment - Data Preprocessing & EDA

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'sri_lanka_vehicles_dataset.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Dataset")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

if 'price_per_km' in df.columns:
    df = df.drop('price_per_km', axis=1)

print(f"Dataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("\nColumn Names:", list(df.columns))
print("\nFirst 5 Rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2. BASIC INFO
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Dataset Information")
print("=" * 60)
print("\nData Types:")
print(df.dtypes)
print("\nBasic Statistics:")
print(df.describe())

# ─────────────────────────────────────────────
# 3. MISSING VALUES
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Missing Values")
print("=" * 60)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
print(missing_df[missing_df['Missing Count'] > 0])
if missing_df['Missing Count'].sum() == 0:
    print("No missing values found. Dataset is clean.")

# ─────────────────────────────────────────────
# 4. DUPLICATE CHECK
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Duplicate Rows")
print("=" * 60)
dups = df.duplicated().sum()
print(f"Duplicate Rows: {dups}")
if dups > 0:
    df = df.drop_duplicates()
    print(f"Duplicates removed. New shape: {df.shape}")

# ─────────────────────────────────────────────
# 5. TARGET VARIABLE ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Target Variable (price) Analysis")
print("=" * 60)
print(f"Price Range: {df['price'].min():,.0f} - {df['price'].max():,.0f} LKR")
print(f"Mean Price: {df['price'].mean():,.0f} LKR")
print(f"Median Price: {df['price'].median():,.0f} LKR")
print(f"Std Dev: {df['price'].std():,.0f} LKR")
print(f"Skewness: {df['price'].skew():.4f}")

# ─────────────────────────────────────────────
# 6. CATEGORICAL FEATURE DISTRIBUTIONS
# ─────────────────────────────────────────────
cat_cols = ['brand', 'fuel_type', 'transmission', 'body_type',
            'location', 'condition', 'seller_type', 'brand_category',
            'fuel_efficiency', 'tax_era']

print("\n" + "=" * 60)
print("STEP 6: Categorical Feature Value Counts")
print("=" * 60)
for col in cat_cols:
    print(f"\n{col}:\n{df[col].value_counts().to_string()}")

# ─────────────────────────────────────────────
# 7. OUTLIER DETECTION (IQR)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Outlier Detection (Price & Mileage)")
print("=" * 60)

def detect_outliers_iqr(series, name):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = series[(series < lower) | (series > upper)]
    print(f"{name}: Q1={Q1:.0f}, Q3={Q3:.0f}, IQR={IQR:.0f}, "
          f"Lower Bound={lower:.0f}, Upper Bound={upper:.0f}, "
          f"Outliers={len(outliers)} ({100*len(outliers)/len(series):.1f}%)")
    return lower, upper

p_lower, p_upper = detect_outliers_iqr(df['price'], 'Price')
m_lower, m_upper = detect_outliers_iqr(df['mileage'], 'Mileage')

# Cap outliers using winsorisation
df['price']   = df['price'].clip(lower=p_lower, upper=p_upper)
df['mileage'] = df['mileage'].clip(lower=0,      upper=m_upper)
print("Outliers capped using Winsorisation (IQR method).")

# ─────────────────────────────────────────────
# 8. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Feature Engineering")
print("=" * 60)

# Log-transform price (for model training target)
df['log_price'] = np.log1p(df['price'])
print("✓ Created: log_price (log-transformed target)")

# Age group bins
df['age_group'] = pd.cut(df['vehicle_age'],
                          bins=[0, 3, 7, 12, 100],
                          labels=['New (0-3yr)', 'Mid (4-7yr)',
                                  'Old (8-12yr)', 'Vintage (13+yr)'])
print("✓ Created: age_group (binned vehicle age)")

# Mileage per year - LEGITIMATE FEATURE (no price involved)
df['mileage_per_year'] = df['mileage'] / (df['vehicle_age'].replace(0, 1))
print("✓ Created: mileage_per_year (usage intensity)")

# Mileage category - ordinal encoding based on usage
df['mileage_category'] = pd.cut(df['mileage'],
                                 bins=[0, 50000, 100000, 150000, 300000],
                                 labels=['Low', 'Medium', 'High', 'Very High'])
print("✓ Created: mileage_category (binned mileage)")

# Brand-age interaction (luxury brands depreciate slower)
df['brand_age_interaction'] = (df['brand_category'] == 'Luxury').astype(int) * df['vehicle_age']
print("✓ Created: brand_age_interaction (luxury × age)")

# High-demand model premium (popular models retain value better)
df['is_premium_combo'] = ((df['is_popular_model'] == 1) & 
                           (df['condition'] == 'Reconditioned')).astype(int)
print("✓ Created: is_premium_combo (popular + reconditioned)")

# Engine size category
df['engine_category'] = pd.cut(df['engine_capacity'],
                                bins=[-1, 660, 1500, 2000, 4000],
                                labels=['Micro', 'Small', 'Medium', 'Large'])
print("✓ Created: engine_category (binned engine)")

print(f"\nFinal Dataset Shape: {df.shape}")

# ─────────────────────────────────────────────
# 9. ENCODING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 9: Encoding Categorical Variables")
print("=" * 60)

# Ordinal encoding for ordered categories
fuel_eff_map = {'Very High': 4, 'High': 3, 'Medium': 2, 'Low': 1}
df['fuel_efficiency_enc'] = df['fuel_efficiency'].map(fuel_eff_map)

tax_era_map = {'Low Tax Era': 1, 'Medium Tax Era': 2, 'High Tax Era': 3}
df['tax_era_enc'] = df['tax_era'].map(tax_era_map)

mileage_cat_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
df['mileage_category_enc'] = df['mileage_category'].map(mileage_cat_map)

engine_cat_map = {'Micro': 1, 'Small': 2, 'Medium': 3, 'Large': 4}
df['engine_category_enc'] = df['engine_category'].map(engine_cat_map)

# One-hot encoding for nominal categories
ohe_cols = ['brand', 'fuel_type', 'transmission', 'body_type',
            'condition', 'seller_type', 'brand_category', 'location']

df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=True)
print(f"Encoded shape: {df_encoded.shape}")

# ─────────────────────────────────────────────
# 10. SAVE PREPROCESSED DATA
# ─────────────────────────────────────────────
save_path = os.path.join(OUTPUT_PATH, 'preprocessed_data.csv')
df_encoded.to_csv(save_path, index=False)
print(f"\nPreprocessed data saved: {save_path}")

clean_path = os.path.join(OUTPUT_PATH, 'cleaned_data.csv')
df.to_csv(clean_path, index=False)
print(f"Cleaned data saved: {clean_path}")

# ─────────────────────────────────────────────
# 11. EDA PLOTS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Generating EDA Plots")
print("=" * 60)

plt.rcParams.update({'font.size': 11})

# --- Plot 1: Price Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['price'], bins=50, color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title('Price Distribution (Raw)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Price (LKR)')
axes[0].set_ylabel('Count')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

axes[1].hist(df['log_price'], bins=50, color='coral', edgecolor='white', alpha=0.85)
axes[1].set_title('Price Distribution (Log-Transformed)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('log(Price)')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig01_price_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 2: Price by Brand ---
fig, ax = plt.subplots(figsize=(14, 6))
brand_price = df.groupby('brand')['price'].median().sort_values(ascending=False)
colors = ['#2196F3' if b != 'BMW' else '#FF9800' for b in brand_price.index]
bars = ax.bar(brand_price.index, brand_price.values / 1e6, color=colors, edgecolor='white', linewidth=0.8)
ax.set_title('Median Vehicle Price by Brand', fontsize=14, fontweight='bold')
ax.set_xlabel('Brand')
ax.set_ylabel('Median Price (Million LKR)')
ax.tick_params(axis='x', rotation=45)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig02_price_by_brand.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 3: Price vs Mileage scatter ---
fig, ax = plt.subplots(figsize=(10, 6))
fuel_colors = {'Petrol': '#e74c3c', 'Hybrid': '#2ecc71', 'Diesel': '#3498db', 'Electric': '#9b59b6'}
for fuel, group in df.groupby('fuel_type'):
    ax.scatter(group['mileage'], group['price']/1e6, alpha=0.35, s=18,
               label=fuel, color=fuel_colors.get(fuel, 'grey'))
ax.set_title('Price vs Mileage by Fuel Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Mileage (km)')
ax.set_ylabel('Price (Million LKR)')
ax.legend(title='Fuel Type', framealpha=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig03_price_vs_mileage.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 4: Price by Fuel Type (Box) ---
fig, ax = plt.subplots(figsize=(10, 6))
df.boxplot(column='price', by='fuel_type', ax=ax,
           boxprops=dict(color='steelblue'),
           medianprops=dict(color='red', linewidth=2))
ax.set_title('Price Distribution by Fuel Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Fuel Type')
ax.set_ylabel('Price (LKR)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig04_price_by_fuel.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 5: Price by Vehicle Age ---
fig, ax = plt.subplots(figsize=(10, 6))
age_price = df.groupby('vehicle_age')['price'].median()
ax.plot(age_price.index, age_price.values/1e6, marker='o', color='steelblue', linewidth=2, markersize=5)
ax.fill_between(age_price.index, age_price.values/1e6, alpha=0.15, color='steelblue')
ax.set_title('Median Price vs Vehicle Age', fontsize=14, fontweight='bold')
ax.set_xlabel('Vehicle Age (Years)')
ax.set_ylabel('Median Price (Million LKR)')
ax.invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig05_price_vs_age.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 6: Correlation Heatmap ---
num_cols = ['price', 'mileage', 'engine_capacity', 'features_count',
            'vehicle_age', 'mileage_per_year',
            'fuel_efficiency_enc', 'tax_era_enc', 'brand_age_interaction']
corr_matrix = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(11, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu',
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig06_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 7: Transmission and Condition Counts ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df['transmission'].value_counts().plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'], edgecolor='white')
axes[0].set_title('Transmission Type Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Transmission')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)

df['condition'].value_counts().plot(kind='bar', ax=axes[1], color=['#2ecc71', '#f39c12'], edgecolor='white')
axes[1].set_title('Vehicle Condition Distribution', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Condition')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, 'fig07_transmission_condition.png'), dpi=150, bbox_inches='tight')
plt.close()

print("All EDA plots saved to outputs/")
print("\n✓ Preprocessing complete!")

