# PaySim Fraud Detection — Chain-Aware Real-Time Triage System

Group project for **DATA 245 – Machine Learning Technologies**.

## Team

- Ritika Neema
- Sneha Singh
- Centhurvelan Ramalingam Sakthivel
- Rajesh Paruchuri

## Project description

Financial fraud detection is a rare-event classification problem — fraudulent
transactions make up less than 0.13% of activity in the [PaySim mobile money
simulation dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle).
Rather than scoring each transaction independently, as prior work on this
dataset does, this project's core finding is a structural pattern in how the
fraud actually occurs:

> **97% of all fraud follows a two-step chain — one `TRANSFER` followed by one
> `CASH_OUT` of the exact same amount in the same time step.**

Built on that, the notebook implements a **chain-aware fraud triage system**
that:
1. Detects fraud at the individual transaction level (the standard approach).
2. Additionally detects fraud at the *chain* level (`is_chain_member`,
   `chain_size` engineered as explicit features) — the project's novel
   contribution.
3. Converts calibrated fraud probabilities into a business-facing triage
   decision: **Allow (GREEN) / Review (YELLOW) / Block (RED)**.

## What's in this repo

| File | Contents |
|---|---|
| `01_eda_paysim.ipynb` | Full pipeline: EDA, chain-pattern discovery, preprocessing, an A/B (no-chain vs. chain-aware) benchmark across Logistic Regression / Random Forest / XGBoost / LightGBM / CatBoost / Balanced RF / Gaussian NB, probability calibration, cost-sensitive triage thresholds, SHAP + LIME explainability, drift monitoring (PSI + temporal validation), seed-stability checks, and error analysis |
| `Preprocessing_PaySim.ipynb` | Standalone preprocessing pipeline |
| `Baseline_Models_PaySim.ipynb` | Baseline model benchmarking |
| `PaySim_Fraud_Presentation.pptx` | Project presentation slides |

**Not included:**
- **The raw dataset** (`PS_20174392719_1491204439457_log.csv`, ~534MB) — download it from
  [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it in this
  directory before running `01_eda_paysim.ipynb`.
- **The Streamlit deployment app** (`app.py`, model artifacts, `MODEL_CARD.md`)
  referenced in the notebook's later sections — per the accompanying paper's
  reproducibility statement, the canonical source for the notebook, source
  code, and deployment prototype together is
  [Snehasingh-21/PaySim-Fraud-Triage](https://github.com/Snehasingh-21/PaySim-Fraud-Triage).
  This repo is a notebook-only mirror for the analysis/modeling work.

## Key result

A calibrated CatBoost model (sigmoid calibration) was selected as the final
deployed scorer, achieving PR-AUC ≈ 0.999 on a held-out test split, validated
for stability across 5 random seeds and a temporal (train-on-past,
test-on-future) holdout. See `01_eda_paysim.ipynb` for the full methodology,
ablation study isolating each pipeline layer's contribution, and the
cost-sensitive triage policy.

## Running it

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost shap lime matplotlib seaborn joblib
jupyter notebook 01_eda_paysim.ipynb
```
