<div align="center">

# 🔩 Steel Rotating Bending Fatigue Strength Prediction

### End-to-End ML Pipeline · Hyperparameter Tuning · SHAP Explainability

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-189AB4?style=for-the-badge)](https://xgboost.readthedocs.io)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian%20HPO-8B5CF6?style=for-the-badge)](https://optuna.org)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-EF4444?style=for-the-badge)](https://shap.readthedocs.io)
[![Colab](https://img.shields.io/badge/Run%20in-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

</div>

---

## 📖 Overview

This project builds and evaluates **11 machine learning regression models** to predict the **rotating bending fatigue strength** (MPa at 10⁷ cycles) of steel alloys. The dataset contains ~500 samples described by chemical composition and heat-treatment processing parameters.

Beyond basic model training, the pipeline includes **Bayesian hyperparameter optimisation with Optuna** and a **complete SHAP explainability suite** — making every prediction transparent and interpretable.

---

## 📁 Repository Structure

```
steel-fatigue-prediction/
│
├── Steel_Fatigue_Prediction.ipynb   ← Main notebook
├── fatigue_data_.csv                ← Dataset (~500 steel samples)
├── requirements.txt                 ← Python dependencies
├── LICENSE
└── README.md
```

---

## 🗂️ Dataset

| Property | Detail |
|----------|--------|
| Samples | ~500 steel specimens |
| Features | 25 (16 processing + 9 chemical) |
| Target | `Fatigue` — Rotating Bending Fatigue Strength (MPa, 10⁷ cycles) |
| Target range | ~208 MPa → ~1,190 MPa |

**Processing Parameters (16):** `NT`, `THT`, `THt`, `THQCr`, `CT`, `Ct`, `DT`, `Dt`, `QmT`, `TT`, `Tt`, `TCr`, `RedRatio`, `dA`, `dB`, `dC`

**Chemical Composition (9):** `C`, `Si`, `Mn`, `P`, `S`, `Ni`, `Cr`, `Cu`, `Mo` *(all in wt%)*

---

## 🤖 Models

| # | Model | Category | Input |
|---|-------|----------|-------|
| 1 | Random Forest | Ensemble Tree | Raw |
| 2 | Gradient Boosting | Ensemble Tree | Raw |
| 3 | XGBoost | Ensemble Tree | Raw |
| 4 | ANN (MLP) | Neural Network | Standardised |
| 5 | Bayesian Ridge | Bayesian | Standardised |
| 6 | Lasso | Regularised Linear | Standardised |
| 7 | Ridge | Regularised Linear | Standardised |
| 8 | Linear Regression | Linear | Standardised |
| 9 | Decision Tree | Tree | Raw |
| 10 | KNN | Instance-Based | Standardised |
| 11 | SVR | Kernel | Standardised |

---

## 📊 Pipeline — 14 Sections

```
Section 1  --  Environment Setup           imports, plot style, version check
Section 2  --  Data Loading                3 options: upload / Drive / local
Section 3  --  EDA                         schema, missing values, distributions
Section 4  --  Correlation Heatmap         Pearson r matrix + fatigue bar chart
Section 5  --  K-Means + PCA              elbow method, cluster & fatigue plots
Section 6  --  Data Preparation            train/test 80/20, StandardScaler
Section 7  --  Model Training              all 11 models, 5-fold CV R²
Section 8  --  Model Comparison            leaderboard, bar charts, scatter grid
Section 9  --  Best Model Deep Dive        residuals, ±2σ bands, histogram
Section 10 --  Feature Importance          built-in tree importances
Section 11 --  Hyperparameter Tuning ★     Optuna TPE, 80 trials, history + param importance
Section 12 --  SHAP Explainability ★       beeswarm, bar, dependence, heatmap, waterfall (5 plots)
Section 13 --  Predict New Sample          all models + tuned + SHAP waterfall for new input
Section 14 --  Save All Outputs            CSV + 17 PNG figures auto-downloaded
```

---

## 🎯 Section 11 — Hyperparameter Tuning with Optuna

Uses **TPE (Tree-structured Parzen Estimator)** — a Bayesian sampler that learns from previous trials to propose smarter candidates, far more efficient than GridSearchCV over large spaces.

The best model from Section 7 is automatically selected and tuned. After optimisation, a comparison table and two plots show the improvement:

- `10_optuna_tuning.png` — trial history curve + baseline vs tuned R² bars
- `11_optuna_param_importance.png` — which params mattered most (Fanova)

**Example XGBoost search space:**

| Hyperparameter | Range |
|---------------|-------|
| `n_estimators` | 100 – 800 |
| `learning_rate` | 0.001 – 0.3 (log) |
| `max_depth` | 3 – 10 |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `reg_alpha / reg_lambda` | 1e-8 – 10 (log) |
| `min_child_weight` | 1 – 10 |
| `gamma` | 0 – 5 |

---

## 🔬 Section 12 — SHAP Explainability

SHAP (SHapley Additive exPlanations) gives each feature a **fair, signed contribution** to every individual prediction — showing not just *what* matters but *which direction* it pushes the output.

| Plot | File | What it answers |
|------|------|-----------------|
| Beeswarm | `12_shap_beeswarm.png` | Global ranking + direction + magnitude for every sample |
| Bar chart | `13_shap_bar.png` | Clean ranking by mean absolute SHAP |
| Dependence | `14_shap_dependence.png` | How top 3 features interact with fatigue |
| Heatmap | `15_shap_heatmap.png` | SHAP contributions across all test samples |
| Waterfall | `16_shap_waterfall_*.png` | Why this specific value was predicted |

Section 13 also generates a **SHAP waterfall for any custom new sample**, so every prediction you make comes with a full explanation.

---

## 📤 All Output Files

| File | Description |
|------|-------------|
| `model_results.csv` | Performance table for all 11 models |
| `01_fatigue_distribution.png` | Target variable histogram + boxplot |
| `02_feature_distributions.png` | Grid of all 25 feature distributions |
| `03_correlation_heatmap.png` | Full Pearson correlation matrix |
| `04_elbow_method.png` | K-Means elbow curve |
| `05_kmeans_pca.png` | Clusters and fatigue gradient in PCA space |
| `06_model_comparison.png` | R², RMSE, CV R² bars for all models |
| `07_all_models_actual_vs_predicted.png` | 3×4 scatter grid — all 11 models |
| `08_best_model_analysis.png` | Residual plot, histogram, ±2σ |
| `09_feature_importance.png` | Built-in tree feature importances |
| `10_optuna_tuning.png` | Trial history + baseline vs tuned R² |
| `11_optuna_param_importance.png` | Hyperparameter sensitivity (Fanova) |
| `12_shap_beeswarm.png` | Global SHAP — beeswarm |
| `13_shap_bar.png` | Global SHAP — bar chart |
| `14_shap_dependence.png` | Dependence plots — top 3 features |
| `15_shap_heatmap.png` | SHAP heatmap across test samples |
| `16_shap_waterfall_*.png` | Local SHAP — median + worst-error samples |
| `17_shap_new_sample.png` | Local SHAP — your custom new prediction |

---

## 🚀 Quick Start

### Google Colab (recommended)

1. Open `Steel_Fatigue_Prediction.ipynb` in Colab
2. **Section 1** auto-installs `xgboost`, `optuna`, `shap`
3. **Section 2** — upload `fatigue_data_.csv` when prompted
4. Run all cells — all figures auto-download at the end

> ⏱️ Estimated runtime: 8–15 min on Colab CPU (Optuna 80 trials is the longest step)

### Local

```bash
git clone https://github.com/<your-username>/steel-fatigue-prediction.git
cd steel-fatigue-prediction
pip install -r requirements.txt
jupyter notebook Steel_Fatigue_Prediction.ipynb
```

In **Section 2**, switch from Option A to Option C (local path).

---

## 📦 requirements.txt

```
numpy>=1.23.0
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
optuna>=3.0.0
shap>=0.42.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

---

## 📈 Expected Results

| Rank | Model | R² (Test) |
|------|-------|-----------|
| 🥇 Tuned (Optuna) | XGBoost / GBM | ~0.990+ |
| 🥈 Baseline best | XGBoost / GBM | ~0.985+ |
| 🥉 | Random Forest | ~0.975+ |
| Others | Linear / KNN / SVR | 0.94 – 0.97 |

---

## 🤝 Contributing

Ideas for extension: Streamlit web app · SHAP interaction values · multi-objective optimisation · additional steel datasets

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">Made for materials science + machine learning</div>
