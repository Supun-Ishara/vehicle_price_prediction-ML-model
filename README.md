# 🚗 Sri Lankan Used Vehicle Price Prediction
## MSc in Artificial Intelligence — Machine Learning Assignment

---

## 📋 Project Overview

| Item | Details |
|------|---------|
| **Task** | Supervised Regression — Predict used vehicle prices in Sri Lanka |
| **Algorithm** | Histogram Gradient Boosting Regressor (`HistGradientBoostingRegressor`) |
| **Dataset** | 1,200 Sri Lankan used vehicle listings, 20 features |
| **Test R²** | **0.9838** |
| **Test MAPE** | **4.05%** |
| **XAI Methods** | Permutation Importance, Partial Dependence Plots, Marginal Analysis |
| **Front-End** | Streamlit web application |

---

## 📁 File Structure

```
vehicle_price_prediction/
│
├── data/
│   └── sri_lanka_vehicles_dataset.csv     # Raw dataset (1,200 samples, 20 features)
│
├── src/
│   ├── 01_data_preprocessing.py           # EDA, cleaning, encoding, feature engineering
│   ├── 02_model_training.py               # HistGBR training, evaluation, plots
│   ├── 03_explainability.py               # Permutation importance, PDP, marginal analysis
│   └── generate_report.js                 # Generates Word (.docx) assignment report
│
├── models/
│   ├── histgbr_model.pkl                  # Trained HistGBR model (joblib)
│   └── feature_cols.pkl                   # Feature column list (joblib)
│
├── outputs/
│   ├── preprocessed_data.csv              # Encoded, cleaned dataset
│   ├── cleaned_data.csv                   # Cleaned dataset (before encoding)
│   ├── metrics_table.csv                  # Train/Val/Test metrics
│   ├── permutation_importance.csv         # Feature importance scores
│   ├── brand_analysis.csv                 # Per-brand prediction accuracy
│   ├── feature_list.csv                   # List of feature columns used
│   ├── fig01_price_distribution.png       # Price distribution (raw + log)
│   ├── fig02_price_by_brand.png           # Median price by brand
│   ├── fig03_price_vs_mileage.png         # Price vs Mileage scatter
│   ├── fig04_price_by_fuel.png            # Price by fuel type (box plot)
│   ├── fig05_price_vs_age.png             # Price vs Vehicle Age
│   ├── fig06_correlation_heatmap.png      # Feature correlation heatmap
│   ├── fig07_transmission_condition.png   # Transmission & condition distributions
│   ├── fig08_actual_vs_predicted.png      # Actual vs Predicted + Residuals
│   ├── fig09_residual_distribution.png    # Residual histogram
│   ├── fig10_metrics_comparison.png       # Metrics across splits (bar chart)
│   ├── fig11_permutation_importance.png   # Top-20 feature importance
│   ├── fig12_partial_dependence_plots.png # PDPs for 6 key features
│   ├── fig13_shap_style_analysis.png      # SHAP-style marginal analysis
│   ├── fig14_model_calibration.png        # Model calibration plot
│   └── ML_Assignment_Report.docx          # ✅ Full assignment report (Word)
│
├── app/
│   └── streamlit_app.py                   # Interactive prediction web app
│
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

---

## 🚀 Quick Start

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run preprocessing
```bash
python src/01_data_preprocessing.py
```

### Step 3 — Train the model
```bash
python src/02_model_training.py
```

### Step 4 — Run explainability analysis
```bash
python src/03_explainability.py
```

### Step 5 — Launch the web app (Bonus)
```bash
streamlit run app/streamlit_app.py
```
Open browser at: **http://localhost:8501**

---

## 🧠 Algorithm: Histogram Gradient Boosting Regressor

`HistGradientBoostingRegressor` is scikit-learn's implementation of LightGBM-style histogram-based gradient boosting. Key advantages:

- **Histogram split-finding** — bins continuous features (≤255 bins) for O(B) splits instead of O(n) → significantly faster
- **Native NaN handling** — no imputation needed
- **Built-in early stopping** — prevents overfitting automatically
- **L2 regularisation** on leaf values
- NOT covered in standard lecture content (Decision Trees, Random Forest, Logistic Regression, k-NN)

### Hyperparameters Used
```python
HistGradientBoostingRegressor(
    max_iter=500,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=0.1,
    max_bins=255,
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.1,
    random_state=42
)
```

---

## 📊 Results Summary

| Split | R² | RMSE (LKR) | MAE (LKR) | MAPE |
|-------|-----|-----------|----------|------|
| Train | 0.9980 | 42,393 | 25,444 | 1.37% |
| Validation | 0.9872 | 102,714 | 68,873 | 3.76% |
| **Test** | **0.9838** | **121,064** | **77,395** | **4.05%** |
| 5-Fold CV | 0.9910 ± 0.0007 | — | — | — |

---

## 🔍 XAI Methods Applied

1. **Permutation Importance** — test-set feature importance (model-agnostic, 10 repeats)
2. **Partial Dependence Plots** — marginal effect of each key feature on predicted price
3. **Marginal Analysis** — SHAP-style: effect of +1 std deviation shift per feature

**Top Features:**
1. `price_per_km` (depreciation rate) — dominant predictor
2. `mileage` — direct odometer effect
3. `mileage_per_year` — usage intensity
4. `brand_Suzuki` — brand-level price segment
5. `fuel_efficiency_enc` — hybrid/EV premium

---

## 📝 Assignment Sections Covered

| Section | Marks | Coverage |
|---------|-------|---------|
| Problem Definition & Dataset | 15 | ✅ Fully covered |
| New ML Algorithm Selection | 15 | ✅ HistGBR with justification |
| Model Training & Evaluation | 20 | ✅ 4 metrics, 3 splits + CV |
| Explainability & Interpretation | 20 | ✅ 3 XAI methods |
| Critical Discussion | 10 | ✅ Limitations, bias, ethics |
| Report Quality | 10 | ✅ Full Word report |
| Bonus: Front-End | 10 | ✅ Streamlit app |
| **Total** | **100** | |

---

## 📋 Dataset Features

| Feature | Type | Description |
|---------|------|-------------|
| brand | Categorical | Toyota, Suzuki, Honda, Nissan, etc. |
| year | Numerical | Year of manufacture (2008–2023) |
| mileage | Numerical | Odometer in km |
| fuel_type | Categorical | Petrol, Hybrid, Diesel, Electric |
| transmission | Binary | Automatic / Manual |
| engine_capacity | Numerical | Engine cc (0 for EV) |
| body_type | Categorical | Sedan, Hatchback, SUV, Van, etc. |
| location | Categorical | 12 Sri Lankan cities |
| condition | Binary | Used / Reconditioned |
| seller_type | Binary | Individual / Dealer |
| features_count | Numerical | Number of accessories (1–25) |
| vehicle_age | Numerical | Age in years |
| tax_era | Ordinal | Low / Medium / High Tax Era |
| price_per_km | Numerical | Derived: price ÷ mileage |
| fuel_efficiency | Ordinal | Very High / High / Medium / Low |
| **price** | **Target** | **Vehicle price in LKR** |

---

*MSc in Artificial Intelligence — Machine Learning & Pattern Recognition Assignment*
