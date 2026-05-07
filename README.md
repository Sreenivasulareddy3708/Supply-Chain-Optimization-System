# Supply Chain Optimization System

An integrated, data-driven supply chain decision framework that combines demand forecasting, inventory optimization, and supplier intelligence into a single interactive Streamlit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Inventory Optimization](#inventory-optimization)
- [Supplier Recommendation](#supplier-recommendation)
- [Dashboard](#dashboard)
- [Results](#results)
- [Tech Stack](#tech-stack)

---

## Overview

Traditional supply chain systems suffer from siloed decision-making, reliance on manual planning, and static forecasting models that fail under volatile market conditions. This project addresses these limitations by building an **adaptive, integrated, and intelligent supply chain optimization system** that:

- Predicts future demand using an **Error-Adaptive Ensemble** combining PatchTST (Transformer-based time-series) and XGBoost (residual correction)
- Computes optimal inventory levels using **Safety Stock**, **Reorder Point (ROP)**, and **Reorder Quantity (ROQ)** calculations
- Ranks and recommends suppliers using a **multi-criteria KPI scoring model** based on lead time, cost, reliability, and quality
- Presents all insights in a unified **Streamlit interactive dashboard**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Raw Data Layer                          │
│   Historical Sales │ Inventory │ Supplier KPIs │ Shipments  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Data Preprocessing Pipeline                    │
│        Cleaning │ Validation │ Feature Engineering          │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐
│   Demand    │  │  Inventory   │  │      Supplier        │
│ Forecasting │  │ Optimization │  │   Recommendation     │
│  Module     │  │   Module     │  │      Engine          │
│             │  │              │  │                      │
│ PatchTST +  │  │ Safety Stock │  │ Multi-Criteria KPI   │
│  XGBoost    │  │  ROP │ ROQ   │  │   Scoring & Ranking  │
│  Ensemble   │  │   EOQ        │  │                      │
└──────┬──────┘  └──────┬───────┘  └──────────┬───────────┘
       │                │                     │
       └────────────────┼─────────────────────┘
                        ▼
          ┌─────────────────────────┐
          │   Streamlit Dashboard   │
          │  Integrated Decision UI │
          └─────────────────────────┘
```

---

## Project Structure

```
Supply chain management/
│
├── dashboard_app.py              # Main Streamlit dashboard (entry point)
├── demand_forecaster.py          # Demand forecasting using Prophet
├── inventory_optimizer.py        # Inventory metrics: Safety Stock, ROP, ROQ
├── supplier_recommendation.py    # Supplier scoring and recommendation engine
├── ensemble.py                   # Error-Adaptive Ensemble (PatchTST + XGBoost)
├── models.py                     # PatchTST and ResidualXGBoost model definitions
├── features.py                   # Feature engineering for XGBoost input
├── config.py                     # Hyperparameters for PatchTST and XGBoost
├── data_generator.py             # Synthetic data generator for testing
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
│
├── data/                         # Raw supply chain datasets
│   ├── demand.csv                # Historical daily sales (3 years, 2022–2024)
│   ├── fact_purchases.csv        # Historical supplier purchase orders
│   ├── inventory.csv             # Current stock levels by product and location
│   ├── dim_products.csv          # Product master data (100 products)
│   ├── dim_suppliers.csv         # Supplier master data (10 suppliers)
│   ├── dim_locations.csv         # Location master data (20 locations)
│   ├── shipments.csv             # Shipment history
│   ├── suppliers.csv             # Supplier list
│   ├── locations.csv             # Location list
│   └── products.csv              # Product list
│
├── dashboard_data/               # Pre-processed outputs consumed by dashboard
│   ├── fact_demand_forecasts.csv # 90-day demand forecasts with confidence intervals
│   ├── inventory_insights.csv    # Safety stock, ROP, ROQ, and inventory status
│   └── metrics_forecast_accuracy.csv  # MAE, RMSE, MAPE per product-location
│
└── reports/                      # Experiment results and analysis artifacts
    ├── 01_train_test_split.png
    ├── 02_actual_vs_forecast_optimized.png
    ├── Ablation study results.png
    ├── metrics_summary.csv
    ├── metrics_summary_optimized.csv
    └── comparison/               # Baseline vs proposed model comparison scripts
        ├── 01_data_preparation.py
        ├── 02_baseline_models.py
        ├── 02b_proposed_ensemble.py
        ├── 03_metrics_calculator.py
        ├── run_complete_experiment.py
        ├── TABLE_I_Overall_Performance.csv
        └── TABLE_II_Volatility_Analysis.csv
```

---

## Modules

### `demand_forecaster.py` — Demand Forecasting
Uses **Facebook Prophet** to generate 90-day demand forecasts for every product-location combination.

- Loads historical daily sales data and applies robust NaN and type handling
- Performs train/validation split (last 30 days held out for accuracy evaluation)
- Fits Prophet with yearly and weekly seasonality enabled
- Calculates **MAE**, **RMSE**, and **MAPE** on the validation set
- Outputs forecasts with confidence intervals (`yhat_lower`, `yhat_upper`)
- Saves results to `dashboard_data/fact_demand_forecasts.csv`

### `inventory_optimizer.py` — Inventory Optimization
Computes optimal inventory decisions for each product-location pair based on historical demand and forecasts.

| Metric | Formula |
|---|---|
| Safety Stock | `Z × σ_daily × √(Lead Time)` |
| Reorder Point | `(Avg Daily Demand × Lead Time) + Safety Stock` |
| Reorder Quantity | Forecasted demand over the reorder period (next 30 days) |

- Default service level: **95%** (Z-score = 1.645)
- Lead time sourced from supplier master data
- Inventory status classified as: `Optimal`, `Reorder Needed`, `Critical`, `Out of Stock`, `Potential Overstock`
- Saves results to `dashboard_data/inventory_insights.csv`

### `supplier_recommendation.py` — Supplier Intelligence
Evaluates and ranks suppliers using a **multi-criteria scoring model** based on historical purchase data.

Scoring criteria:
- **On-time delivery rate** — percentage of orders delivered on time
- **Average actual lead time** — computed from order and delivery dates
- **Cost competitiveness** — average unit price relative to other suppliers
- **Quality / reliability** — derived from purchase history KPIs

Outputs ranked supplier recommendations per product to support procurement decisions.

### `ensemble.py` + `models.py` — Error-Adaptive Ensemble
Implements the research-level forecasting model combining two complementary components:

**Component 1 — PatchTST (Trend Modeling)**
- Transformer-based time-series model
- Divides the input sequence into patches (patch length: 16, stride: 16)
- Uses a 3-layer Transformer encoder with 4 attention heads and `d_model=128`
- Forecast horizon: 90 days

**Component 2 — ResidualXGBoost (Spike Correction)**
- Trained on the residuals (errors) of PatchTST predictions
- Captures non-linear demand patterns and volatility spikes missed by the Transformer
- Features include: lag features (1, 7, 14, 30 days), rolling statistics (7 and 30-day windows), temporal features (day of week, month, quarter)

**Adaptive Weighting**
- Final forecast = `w_patch × PatchTST_output + w_xgb × XGBoost_residual`
- Weights computed via inverse-error weighting using RMSE and MAE on a validation set
- Ensures dynamic balance between trend stability and spike correction

### `features.py` — Feature Engineering
Generates the XGBoost feature matrix from raw demand data:
- PatchTST trend forecast as a feature
- Temporal features: day of week, day of month, month, quarter, is_weekend
- Lag features: 1, 7, 14, 30 days
- Rolling statistics: 7-day and 30-day mean, std, max, min
- Differential features: 1-day and 7-day demand differences

### `config.py` — Hyperparameter Configuration

| Model | Key Parameters |
|---|---|
| PatchTST | `patch_len=16`, `stride=16`, `d_model=128`, `n_heads=4`, `e_layers=3`, `d_ff=256`, `dropout=0.1`, `lr=1e-4`, `epochs=100` |
| XGBoost | `n_estimators=200`, `max_depth=5`, `lr=0.05`, `subsample=0.8`, `early_stopping=20` |

---

## Dataset

| File | Description | Size |
|---|---|---|
| `demand.csv` | Daily sales per product-location (2022–2024) | ~109 MB |
| `fact_purchases.csv` | Historical purchase orders with delivery dates | ~5.2 MB |
| `inventory.csv` | Current stock on hand by product and location | ~49 KB |
| `dim_products.csv` | 100 product definitions | ~3.7 KB |
| `dim_suppliers.csv` | 10 supplier definitions with lead times | ~1.4 KB |
| `dim_locations.csv` | 20 location definitions | ~0.6 KB |

> **Note:** `demand.csv` and `fact_purchases.csv` are excluded from this repository due to file size. Run `data_generator.py` to generate synthetic equivalents for local testing.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/supply-chain-optimization.git
cd supply-chain-optimization

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data (if demand.csv is not present)
python data_generator.py
```

---

## Usage

### Step 1 — Run Demand Forecasting
```bash
python demand_forecaster.py
```
Generates `dashboard_data/fact_demand_forecasts.csv` and `metrics_forecast_accuracy.csv`.

### Step 2 — Run Inventory Optimization
```bash
python inventory_optimizer.py
```
Generates `dashboard_data/inventory_insights.csv`.

### Step 3 — Launch the Dashboard
```bash
streamlit run dashboard_app.py
```
Opens the interactive supply chain dashboard at `http://localhost:8501`.

---

## Dashboard

The Streamlit dashboard provides:
- **Demand Forecast View** — 90-day forecasts with confidence intervals, filterable by product and location
- **Inventory Status View** — Safety stock, ROP, ROQ, and current stock status per product-location
- **Supplier Recommendation View** — Ranked supplier list with KPI scores for procurement decisions
- **Forecast Accuracy Metrics** — MAE, RMSE, MAPE summaries across all product-location combinations

---

## Results

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Baseline Prophet | — | — | — |
| PatchTST only | — | — | — |
| **PatchTST + XGBoost Ensemble** | **Best** | **Best** | **Best** |

> Detailed comparison results are available in `reports/comparison/TABLE_I_Overall_Performance.csv` and `TABLE_II_Volatility_Analysis.csv`.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Forecasting | Facebook Prophet, XGBoost, PyTorch (PatchTST) |
| Optimization | SciPy, NumPy |
| Dashboard | Streamlit, Plotly |
| Data Processing | Pandas |
| Version Control | Git |
