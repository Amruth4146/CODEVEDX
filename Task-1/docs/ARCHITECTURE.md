# Architecture & Design: Electricity Consumption Predictor

## 1. Overview
The **CODEVEDX Task-1 Utility Prediction** project is a modular, console-based Machine Learning application built with **Python 3.11+**, **scikit-learn**, **pandas**, **NumPy**, and **joblib**. It forecasts daily electricity consumption in kilowatt-hours (**kWh**) from chronological date records and historical usage patterns.

---

## 2. Project Directory Structure
```
d:/CODEVEDEX/Task-1/
│
├── .gitignore                    # Version control ignore definitions
├── requirements.txt              # Production and testing dependencies
├── README.md                     # High-level project documentation and quickstart
├── main.py                       # Console application entrypoint & CLI dispatcher
│
├── src/                          # Core business logic and ML modules
│   ├── __init__.py               # Package marker
│   ├── data_manager.py           # Dataset I/O, validation, feature engineering
│   ├── predictor.py              # Regressor wrapper, training, evaluation, inference
│   └── utils.py                  # Project paths, feature schemas, console formatting
│
├── data/                         # Historical usage datasets
│   └── usage_data.csv            # Daily electricity consumption records (date, usage_kwh)
│
├── models/                       # Serialized model artifacts
│   └── electricity_model.joblib  # Trained model artifact with metadata
│
├── docs/                         # Project documentation
│   ├── ARCHITECTURE.md           # System architecture and design documentation
│   └── USAGE_GUIDE.md            # Windows PowerShell execution instructions
│
└── tests/                        # Automated unit tests
    ├── __init__.py
    ├── test_data_manager.py      # Tests for validation, I/O, and preprocessing
    ├── test_predictor.py         # Tests for ML model training, saving, loading
    └── test_utils.py             # Tests for console helpers and schemas
```

---

## 3. Data Pipeline & Management (`src/data_manager.py`)

### 3.1 Schema Definition
Electricity consumption is stored in `data/usage_data.csv` with two core columns:
- `date`: Calendar date in ISO 8601 format (`YYYY-MM-DD`).
- `usage_kwh`: Daily electricity consumed in kilowatt-hours (finite, non-negative `float`).

### 3.2 Data Validation & Integrity
1. **Strict ISO Date Validation (`validate_iso_date`)**:
   - Ensures string format matches `YYYY-MM-DD`.
   - Validates calendar dates (e.g., handles leap years like `2024-02-29` and rejects non-existent dates like `2023-02-29` or `2024-04-31`).
2. **Usage Domain Validation (`validate_usage_value`)**:
   - Requires finite, non-negative numbers (`>= 0.0 kWh`).
   - Rejects `NaN`, `+Infinity`, `-Infinity`, and negative values.
3. **Duplicate Prevention**:
   - Checks if a date already exists when adding records. Recommends `update_record()` for modifications.
4. **Chronological Sorting & Overwrite Safety**:
   - All records are sorted by date prior to saving.
   - `save_data(..., allow_overwrite=False)` guards against unintentional file overwrites.
5. **Demonstration Synthetic Dataset (`generate_synthetic_sample_data`)**:
   - Generates realistic 2-year synthetic electricity usage incorporating base demand, seasonal temperature effects, weekend spikes, and random variance.
   - Explicitly tagged with the disclaimer: *"DEMONSTRATION ONLY: Synthetic records generated for modeling demonstration. Never present or treat synthetic data as actual utility consumption."*

---

## 4. Feature Engineering Architecture

To capture temporal seasonality and short-term autocorrelation, the dataset is transformed into supervised ML features:

| Feature Name | Category | Description |
| :--- | :--- | :--- |
| `day_of_week` | Calendar | Integer `0` (Monday) through `6` (Sunday) |
| `day_of_month` | Calendar | Integer `1` through `31` |
| `month` | Calendar | Integer `1` through `12` |
| `quarter` | Calendar | Integer `1` through `4` |
| `is_weekend` | Calendar | Binary indicator (`1` if Saturday/Sunday, else `0`) |
| `day_of_year` | Calendar | Day ordinal `1` through `366` |
| `lag_1_usage` | Usage History | Previous day's electricity consumption ($t-1$) |
| `lag_7_usage` | Usage History | Same day last week's consumption ($t-7$) |
| `rolling_7_mean` | Usage History | Trailing 7-day average consumption |

---

## 5. Machine Learning Predictor (`src/predictor.py`)

- **Algorithm**: `RandomForestRegressor` (scikit-learn) with 100 estimators, configured for high generalization and robustness against non-linear patterns.
- **Evaluation Metrics**:
  - **Mean Absolute Error (MAE)**: Average magnitude of forecast errors in kWh.
  - **Root Mean Squared Error (RMSE)**: Penalizes larger prediction deviations.
  - **$R^2$ Score**: Coefficient of determination representing the variance explained by the model.
  - **Mean Absolute Percentage Error (MAPE)**: Relative error percentage.
- **Persistence**: Models are serialized using `joblib` into `models/electricity_model.joblib` alongside trained feature names, date of training, and evaluation metrics.
