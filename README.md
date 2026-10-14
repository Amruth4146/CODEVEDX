# CODEVEDX / Task-1-Utility-Prediction

A modular, console-based Machine Learning application in **Python 3.11+** that predicts daily electricity consumption in kilowatt-hours (**kWh**) from historical date and usage records.

Built strictly using standard, battle-tested ML libraries: **pandas**, **NumPy**, **scikit-learn**, and **joblib**, without any external databases, web frameworks (Flask/Streamlit), or paid APIs.

---

## 📌 Features

- **Robust Data Management (`src/data_manager.py`)**:
  - Stores daily records in `data/usage_data.csv` with columns `date`, `usage_kwh`.
  - Full CRUD operations: create/initialize CSV, load, add new record, update record by date, and view records.
  - Strict ISO 8601 date validation (`YYYY-MM-DD`, validating calendar dates & leap years).
  - Strict usage value validation (finite, non-negative float `>= 0.0 kWh`, rejecting `NaN`, `Inf`, or negative numbers).
  - Duplicate date detection and accidental overwrite protection.
  - Chronological ordering enforced on all data I/O.
  - Graceful handling of `FileNotFoundError`, `PermissionError`, empty files, and malformed rows.
  - Generates clearly labeled synthetic sample data for demonstration (with explicit disclaimer: *never present synthetic data as actual consumption*).

- **Feature Engineering Pipeline**:
  - Calendar attributes: `day_of_week`, `day_of_month`, `month`, `quarter`, `is_weekend`, `day_of_year`.
  - Historical usage lags: `lag_1_usage` (yesterday), `lag_7_usage` (same day last week), and `rolling_7_mean` (7-day trailing average).

- **Machine Learning Regressor (`src/predictor.py`)**:
  - Trains a `RandomForestRegressor` model capable of capturing complex seasonal and weekly demand waves.
  - Comprehensive model evaluation: MAE, RMSE, $R^2$, and MAPE.
  - Model serialization and restoration using `joblib` into `models/electricity_model.joblib`.

- **User-Friendly Console Interface (`main.py`)**:
  - Interactive numbered menu for everyday use.
  - CLI argument mode (`--train`, `--summary`, `--predict`, `--add-record`, `--batch-predict`) for automation and testing.
  - Windows PowerShell friendly (CP1252-safe ASCII output, UTF-8 compatibility).

---

## 📂 Project Structure

```
d:/CODEVEDEX/Task-1/
├── .gitignore                    # Git ignore definitions
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
├── main.py                       # CLI entrypoint & interactive menu
│
├── src/                          # Modular source code
│   ├── __init__.py               # Package marker
│   ├── data_manager.py           # Data I/O, validation, feature extraction
│   ├── predictor.py              # ML model pipeline, evaluation, and inference
│   └── utils.py                  # Project constants, paths, and CLI helpers
│
├── data/                         # Dataset storage
│   └── usage_data.csv            # Electricity consumption records (date, usage_kwh)
│
├── models/                       # Model artifact storage
│   └── electricity_model.joblib  # Serialized trained model
│
├── docs/                         # Extended documentation
│   ├── ARCHITECTURE.md           # Architecture, schema, and design patterns
│   └── USAGE_GUIDE.md            # PowerShell command reference & workflows
│
└── tests/                        # Automated unit tests
    ├── __init__.py
    ├── test_data_manager.py      # Data manager tests (validations, I/O)
    ├── test_predictor.py         # Predictor tests (train, save, load, inference)
    └── test_utils.py             # Utility tests
```

---

## 🚀 Quick Start (Windows PowerShell)

### 1. Prerequisites
Ensure you have **Python 3.11+** installed:
```powershell
py -3.11 --version
```

### 2. Set Up Virtual Environment & Dependencies
```powershell
# Navigate to project directory
cd d:\CODEVEDEX\Task-1

# Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# Activate the virtual environment
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Interactive Console Menu
```powershell
python main.py
```
This presents the interactive menu:
```text
=================================================================
  CODEVEDX | Utility Usage Prediction Tool
=================================================================
  1. View usage records
  2. Add usage record
  3. Update usage record
  4. View usage statistics
  5. Train ML model
  6. Predict future usage
  7. View model evaluation
  8. Exit
-----------------------------------------------------------------
Enter your choice [1-8]:
```

### Option B: PowerShell CLI Commands

| Operation | PowerShell Command |
| :--- | :--- |
| **Generate Sample Data** | `python main.py --generate-data` |
| **Dataset Summary** | `python main.py --summary` |
| **View Latest Records** | `python main.py --view-records 10` |
| **Add New Record** | `python main.py --add-record 2025-01-01 24.50` |
| **Update Existing Record** | `python main.py --update-record 2025-01-01 26.00` |
| **Train ML Model** | `python main.py --train` |
| **View Model Info** | `python main.py --model-info` |
| **Single Day Predict** | `python main.py --predict --date 2025-07-15 --lag1 26.5 --lag7 25.0 --rolling 25.8` |
| **Batch Forecast** | `python main.py --batch-predict 7` |

---

## 📊 Model Evaluation Results

On the chronological test split (unseen evaluation data):
- **Mean Absolute Error (MAE)**: `~1.65 kWh`
- **Root Mean Squared Error (RMSE)**: `~2.07 kWh`
- **$R^2$ Score**: `~0.726` (strong variance explanation)
- **Mean Absolute Percentage Error (MAPE)**: `~6.51%`

---

## 🧪 Running Automated Tests

Run the test suite using `pytest`:
```powershell
pytest -v
```
All 30 unit tests pass, verifying data loading, validation, record mutation, lag generation, model training, persistence, and console helpers.

---

## 📋 Code Quality & PEP8 Compliance
- Fully PEP8 compliant with clean formatting, descriptive variable naming, and type annotations (`typing.Union`, `Optional`, `Tuple`, etc.).
- Modular separation of concerns across `data_manager.py`, `predictor.py`, and `utils.py`.
- No extraneous dependencies or future features outside the project scope.
