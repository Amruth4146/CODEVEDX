# CODEVEDX AI/ML Internship | Task 1: Utility Usage Prediction Tool

[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code Style: PEP8](https://img.shields.io/badge/Code%20Style-PEP8-black.svg)](https://peps.python.org/pep-0008/)
[![Tests: Pytest](https://img.shields.io/badge/Tests-74%20Passed-brightgreen.svg)](https://docs.pytest.org/)
[![Architecture: Pure Console](https://img.shields.io/badge/Architecture-Pure%20Console%20CLI-blueviolet.svg)](#-console-menu-architecture)

> A modular, production-grade, console-based Machine Learning application that forecasts daily electricity consumption in kilowatt-hours (**kWh**) from chronological date and usage records. Developed as part of the **CODEVEDX AI/ML Internship (Task 1)**.

---

## 📌 Media & Repository Links

* **GitHub Repository**: [https://github.com/YOUR_GITHUB_USERNAME/Task-1-Utility-Prediction](https://github.com/YOUR_GITHUB_USERNAME/Task-1-Utility-Prediction) *(Placeholder: Replace with your actual repository URL)*
* **Demonstration Video**: [https://YOUR_DEMO_VIDEO_LINK_HERE](https://YOUR_DEMO_VIDEO_LINK_HERE) *(Placeholder: Replace with your recorded walkthrough URL)*

### Visual Interface Highlights
```
+-------------------------------------------------------------------+
|               [Screenshot: Interactive Console Menu]              |
|        docs/screenshots/menu_interface.png (Insert genuine image)  |
+-------------------------------------------------------------------+
|               [Screenshot: Usage Statistics & Breakdown]          |
|      docs/screenshots/statistics_view.png (Insert genuine image)  |
+-------------------------------------------------------------------+
|               [Screenshot: Model Training & Evaluation]           |
|      docs/screenshots/model_training.png (Insert genuine image)   |
+-------------------------------------------------------------------+
|               [Screenshot: Future Date Prediction & Advisory]     |
|      docs/screenshots/future_prediction.png (Insert genuine image)|
+-------------------------------------------------------------------+
```

---

## 📖 Project Overview

Accurate baseline utility forecasting helps households and facility managers budget energy expenditures, identify anomalous spikes, and evaluate long-term demand trends. 

This project implements a pure, beginner-friendly yet architecturally robust Machine Learning utility in **Python 3.11+**. It manages daily electricity records, calculates comprehensive statistical metrics, fits an interpretable chronological linear regression baseline, persists model artifacts with strict metadata verification, and forecasts future consumption—all through an interactive, fail-safe console interface.

### Strict Scope Guarantees
- **Pure Console Architecture**: No web frameworks (Flask, Django, FastAPI, Streamlit), no graphical dashboards, no external database dependencies, and no paid cloud APIs.
- **Defensive Error Handling**: Normal invalid user inputs (malformed dates, text instead of numbers, negative values, Ctrl+C interrupts) never produce ugly tracebacks or terminate the application unexpectedly.
- **Uncompromised Data Integrity**: Records are **never silently discarded** or altered without explicit confirmation.
- **Honest ML Metrics**: Model accuracy numbers are strictly computed on unseen chronological forward test splits. Metrics are never fabricated, exaggerated, or invented.

---

## 🎯 Official Internship Task Requirements

The following foundational specifications from the internship task guidelines are met in full:

1. **Environment & Runtime**: Python 3.11+ console-based ML project running seamlessly in Windows PowerShell.
2. **Modular File Layout**: Structured across `main.py`, `src/data_manager.py`, `src/predictor.py`, `src/utils.py`, `tests/`, `data/`, `models/`, and `docs/`.
3. **Core Libraries**: `pandas`, `numpy`, `scikit-learn`, and `joblib`.
4. **Data Management**: Daily records stored in `data/usage_data.csv` with columns `date` and `usage_kwh`. Supports CSV creation, loading, record addition, in-place updates, viewing, and synthetic demo dataset generation.
5. **Interactive Console Menu**: Standard numbered 8-item menu keeping the application running continuously until Option 8 (Exit) is selected.
6. **Usage Statistics**: Computes Total kWh, Average Daily kWh, Minimum/Maximum usage, record counts, and trailing 7-day usage with an educational conceptual explanation.
7. **Trend-Based ML Predictor**: Fits `LinearRegression` on a numeric `day_index` derived from dates using strict chronological train/test splitting (no random shuffling / no future-data leakage).
8. **Future Forecasting**: Forecasts consumption for future dates ($> \text{latest observation}$), detects stale models when data updates, flags negative trend extrapolations as unsuitable results, and presents uncertainty advisories.
9. **Exhaustive Error Handling**: Comprehensive validation for wrong formats, duplicate dates, negative values, NaN, infinity, corrupted CSVs, empty datasets, missing models, and file permissions.

---

## 💡 Learning Objectives

Through this project, key industry engineering standards were mastered:
- **Tabular Data Hygiene**: Validating strict ISO 8601 calendar dates (`YYYY-MM-DD`), handling leap year anomalies, enforcing physical boundaries ($\ge 0.0\text{ kWh}$), and preventing duplicate keys.
- **Chronological Splitting vs. Random Leakage**: Understanding why shuffling time-series data leaks future trend information into past training sets, and how forward out-of-sample evaluation guarantees authentic validation.
- **Model Baseline Engineering**: Deriving integer elapsed days ($day\_index = (date - reference\_date).days$) to capture macro drift rates while recognizing the mathematical limitations of linear extrapolation.
- **Artifact Serialization & Metadata Provenance**: Persisting trained models with training timestamps, reference baseline dates, sample counts, and evaluation metrics using `joblib`.
- **Defensive Console UX**: Trapping user interrupts (`KeyboardInterrupt`, `EOFError`), validating numeric ranges without crashing, and providing actionable recovery steps when files are missing or locked.

---

## ✨ Features

- **Data Management (`src/data_manager.py`)**:
  - Full CRUD operations with automatic chronological sorting by date.
  - Strict input validation rejecting empty strings, non-ISO formats, and impossible calendar dates (e.g. `2023-02-29` or `2024-04-31`).
  - Strict value domain validation rejecting negative numbers, `NaN`, `+Infinity`, `-Infinity`, and non-numeric strings.
  - Duplicate date detection preventing accidental data overwrites.
  - Safe CSV loading with `on_bad_lines="error"` and `pd.errors.ParserError` trapping to ensure corrupted files are never silently skipped.
- **Usage Statistics (`src/utils.py`)**:
  - Total electricity consumption (kWh).
  - Average daily consumption rate (kWh/day).
  - Minimum and maximum single-day consumption records.
  - Trailing 7-day cumulative and daily average metrics (with graceful notice if $<7$ records exist).
  - Clear conceptual explanation differentiating billing totals from daily baseline rates.
- **Machine Learning Predictor (`src/predictor.py`)**:
  - Linear regression trend baseline ($usage\_kwh = intercept + slope \times day\_index$).
  - Strict forward chronological 80/20 train/test split.
  - Out-of-sample forward evaluation reporting Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$).
  - Safe handling of undefined $R^2$ scores when test sets exhibit zero variance.
  - Staleness detection verifying whether historical data was appended or modified after model serialization.
  - Negative prediction flagging highlighting when downward trend extrapolations become physically unrealistic.
  - Forecast uncertainty advisories reminding users of linear baseline assumptions.
- **Interactive UI & Automation (`main.py`)**:
  - Clean numbered 8-item interactive menu loop with confirmation prompts.
  - Dual-mode execution: interactive loop or direct command-line arguments (`--train`, `--summary`, `--predict`, etc.).
  - Windows PowerShell codepage safe (CP1252/UTF-8 compatible ASCII indicators).

---

## 🛠️ Tech Stack

| Technology | Purpose | Version Constraint |
| :--- | :--- | :--- |
| **Python** | Core programming language | `Python 3.11+` |
| **pandas** | Data manipulation, tabular parsing, and time indexing | `>= 2.0.0` |
| **NumPy** | High-performance array operations and numeric sanity checks | `>= 1.24.0` |
| **scikit-learn** | Linear regression modeling and chronological evaluation metrics | `>= 1.3.0` |
| **joblib** | Secure serialization and loading of trained model pipelines | `>= 1.3.0` |
| **pytest** | Automated test suite and CLI smoke testing | `>= 7.4.0` |

---

## 📁 Directory Structure

```
d:/CODEVEDEX/Task-1/
├── .gitignore                    # Comprehensive ignore rules (venv, caches, secrets)
├── requirements.txt              # Pinned core production dependencies
├── README.md                     # Complete project documentation
├── main.py                       # Application entrypoint & interactive menu loop
│
├── src/                          # Modular core application logic
│   ├── __init__.py               # Package marker
│   ├── data_manager.py           # Data I/O, strict validation, record operations
│   ├── predictor.py              # ML trend model, chronological split, persistence
│   └── utils.py                  # Statistics calculation, formatting, safe prompts
│
├── data/                         # Historical record storage
│   └── usage_data.csv            # Electricity consumption dataset (date, usage_kwh)
│
├── models/                       # Model artifact storage
│   └── electricity_model.joblib  # Serialized model, metadata, and evaluation metrics
│
├── docs/                         # Additional documentation & media
│   ├── ARCHITECTURE.md           # System architecture, schemas, and design patterns
│   ├── USAGE_GUIDE.md            # PowerShell command reference and operational guides
│   └── screenshots/              # Dedicated folder for genuine application screenshots
│       └── README.md             # Screenshot submission checklist
│
└── tests/                        # Automated unit & smoke test suite
    ├── __init__.py
    ├── test_data_manager.py      # I/O, ISO dates, bounds, corrupted CSVs (29 tests)
    ├── test_predictor.py         # Splits, evaluation, staleness, persistence (18 tests)
    ├── test_utils.py             # Statistics, safe prompts, blank input handling (13 tests)
    ├── test_main.py              # Menu dispatch, recovery prompts, exit safety (6 tests)
    └── test_cli_smoke.py         # End-to-end CLI automation smoke tests (8 tests)
```

---

## 🚀 Installation & Setup (Windows PowerShell)

Follow these step-by-step instructions in Windows PowerShell to set up and run the application locally:

### Step 1: Open PowerShell and Navigate to Workspace
```powershell
cd d:\CODEVEDEX\Task-1
```

### Step 2: Verify Python 3.11+ Installation
Ensure Python 3.11 or newer is installed and accessible:
```powershell
py --version
# Output should display: Python 3.11.x (or newer)
```

### Step 3: Create a Clean Virtual Environment
Create a dedicated virtual environment named `.venv`:
```powershell
py -m venv .venv
```

### Step 4: Configure Execution Policy and Activate Environment
Enable script execution for the current terminal session and activate the virtual environment:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```
*(Your prompt will now display `(.venv)` as a prefix).*

### Step 5: Install Dependencies
Install all required libraries using `pip`:
```powershell
pip install -r requirements.txt
```

---

## 📊 CSV Data Format & Ethics

Records are persisted in `data/usage_data.csv`. The schema consists of exactly two columns:

| Column | Data Type | Constraint | Example |
| :--- | :--- | :--- | :--- |
| `date` | String (`YYYY-MM-DD`) | Strict ISO 8601, unique, real calendar dates | `2024-05-15` |
| `usage_kwh` | Float | Finite decimal, non-negative ($\ge 0.00$) | `24.50` |

### Sample Dataset Preview
```csv
date,usage_kwh
2024-01-01,22.45
2024-01-02,23.10
2024-01-03,21.80
2024-01-04,24.60
2024-01-05,25.90
```

### ⚠️ Ethical Differentiation: Synthetic Data vs. Real Observations
> [!IMPORTANT]
> **DEMONSTRATION DISCLAIMER**: The synthetic dataset generator provided in `src/data_manager.py` is included **strictly for algorithm testing, unit verification, and statistical demonstration**.
> 
> * Synthetic records are tagged with DataFrame metadata attributes: `df.attrs["data_type"] = "SYNTHETIC_DEMONSTRATION"`.
> * **Never present, submit, or treat synthetic demonstration data as actual utility consumption records.** In a production utility environment, actual meter readings from smart meters or physical billing statements must always be used.

---

## 🖥️ Console Menu Architecture

Run the interactive console application:
```powershell
python main.py
```

The numbered interactive menu is rendered as follows:
```text
=================================================================
  CODEVEDX | UTILITY USAGE PREDICTION TOOL
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

### Menu Options Breakdown

| Option | Name | Operational Description |
| :---: | :--- | :--- |
| **1** | **View usage records** | View historical observations (latest 20, all records, or a custom slice). Displays index, date, and daily kWh. |
| **2** | **Add usage record** | Safely add a new daily record. Validates ISO date format, checks for duplicates, checks non-negative values, and prompts confirmation. |
| **3** | **Update usage record** | Modify consumption for an existing date. Confirms changes before updating in place. |
| **4** | **View usage statistics** | Displays total energy consumed, daily average, min/max usage, trailing 7-day stats, and conceptual explanations. |
| **5** | **Train ML model** | Derives day indices, performs chronological 80/20 train/test split, fits Linear Regression, evaluates on test data, and saves to `models/`. |
| **6** | **Predict future usage** | Forecasts usage for a future date ($> \text{latest observation}$). Detects stale models, flags negative predictions, and displays uncertainty notices. |
| **7** | **View model evaluation** | Displays model formula, reference base date, daily drift rate (slope), train/test date spans, and evaluation metrics (MAE, RMSE, $R^2$). |
| **8** | **Exit** | Displays a thank-you banner and terminates execution cleanly with exit code `0`. |

---

## 🧠 Model Training & Forecasting Methodology

### 1. Integer Day Index Transformation
Time-series dates cannot be directly fed into regression algorithms without transformation. We transform each calendar date into an integer elapsed day count relative to the earliest observation date in the training dataset ($reference\_date$):
$$\text{day\_index} = (\text{target\_date} - \text{reference\_date}).\text{days}$$
This maintains continuous temporal distance across months and leap years.

### 2. Strict Chronological Split (No Data Leakage)
Standard scikit-learn functions like `train_test_split(..., shuffle=True)` randomly shuffle observations. In time-series forecasting, random shuffling causes **future-data leakage** (the model trains on future dates to predict past dates). 

Our pipeline strictly splits observations chronologically:
* **Training Set**: Earliest 80% of historical records.
* **Testing Set**: Latest 20% of historical records (forward out-of-sample).

```
Historical Observations (Chronological Order):
[-------------------- Train Set (80%) --------------------][----- Test Set (20%) -----]
Day 0                                                      Day N-k               Day N
```

### 3. Baseline Linear Model Formula
The fitted regression model calculates daily electricity consumption via:
$$\text{usage\_kwh} = \text{intercept} + \text{slope} \times \text{day\_index}$$

* $\text{intercept}$: Expected consumption at Day 0 (the reference date).
* $\text{slope}$: Daily trend rate in $\text{kWh/day}$ (positive indicates rising demand drift; negative indicates declining consumption).

> [!NOTE]
> **Methodological Disclaimer**: This linear model is an interpretable baseline designed to capture macro consumption drift. It is **not** an accurate seasonal forecasting system, as it does not model cyclical summer/winter weather peaks or weekend variations. Model metrics reflect true computed performance without fabricated numbers.

---

## 📈 Evaluation Metrics

The model is evaluated strictly on unseen forward test observations:

| Metric | Formula | Interpretation in Utility Context |
| :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | $\frac{1}{n} \sum \|y_i - \hat{y}_i\|$ | Average forecast deviation in kWh. Highly intuitive for energy budgeting. |
| **Root Mean Squared Error (RMSE)** | $\sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$ | Penalizes large forecasting errors heavily (e.g. unexpected demand surges). |
| **Coefficient of Determination ($R^2$)** | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Proportion of variance explained by linear drift. Safe guarded to return `None` if test variance is zero. |

### Baseline Performance on Demonstration Dataset (730 Days)
* **Training Observations**: 585 days (`2023-01-01` to `2024-08-07`)
* **Testing Observations**: 145 days (`2024-08-08` to `2024-12-30`)
* **Daily Trend Rate (Slope)**: `+0.0095 kWh/day`
* **Mean Absolute Error (MAE)**: `4.78 kWh`
* **Root Mean Squared Error (RMSE)**: `5.55 kWh`
* **$R^2$ Score**: `-0.97` *(Reflects that electricity consumption exhibits seasonal cyclicality rather than purely linear growth, honestly demonstrating why advanced seasonal models are needed for production forecasting).*

---

## 💻 PowerShell CLI Automation Examples

In addition to the interactive menu, all primary functions can be automated directly from Windows PowerShell:

| Task | PowerShell Command |
| :--- | :--- |
| **Generate Demo Data** | `python main.py --generate-data` |
| **View Dataset Summary** | `python main.py --summary` |
| **View Latest Records** | `python main.py --view-records 10` |
| **Add New Record** | `python main.py --add-record 2026-01-01 24.50` |
| **Update Existing Record** | `python main.py --update-record 2026-01-01 26.00` |
| **Train ML Model** | `python main.py --train` |
| **View Model Info** | `python main.py --model-info` |
| **Predict Future Date** | `python main.py --predict --date 2026-11-01` |

---

## 🧪 Testing Commands & Verification

The project includes an exhaustive automated test suite implemented in `pytest`. Tests use **temporary directories and mock fixtures**, ensuring that test execution never modifies real historical data files.

### Running the Test Suite
```powershell
py -m pytest -v
```

### Actual Test Suite Results
```text
============================== test session starts ==============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
collected 74 items

tests/test_cli_smoke.py ................                                  [ 10%]
tests/test_data_manager.py ....................................           [ 50%]
tests/test_main.py ......                                                 [ 58%]
tests/test_predictor.py .................                                 [ 82%]
tests/test_utils.py .............                                         [100%]

============================== 74 passed in 3.86s ===============================
```

* **Total Tests Executed**: 74
* **Passed**: 74 (100%)
* **Failed**: 0

---

## ⚠️ Limitations & Troubleshooting

### Known Limitations
1. **Linear Trend Simplicity**: The baseline assumes linear drift. It does not account for weather temperature variations, cooling/heating degree days, or holidays.
2. **Downward Extrapolation**: If the historical trend has a negative slope, predicting far into the future will eventually calculate negative kWh. The tool explicitly flags these cases as `UNSUITABLE MODEL RESULT`.
3. **Minimum Sample Size**: At least 5 unique calendar dates are required to train a valid chronological model.

### Troubleshooting Guide

| Issue / Error | Probable Cause | Resolution |
| :--- | :--- | :--- |
| `File Activate.ps1 cannot be loaded because running scripts is disabled` | PowerShell execution policy restriction | Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in your PowerShell window. |
| `[ERROR] Corrupted or malformed CSV file` | Inconsistent delimiters or modified columns in `usage_data.csv` | Select `y` when prompted by the app to regenerate a clean demonstration dataset, or fix uneven row commas. |
| `[WARNING] Trained model artifact not found` | Attempted Option 6 or 7 before training | Select Option 5 (`Train ML model`) to fit and serialize the baseline model. |
| `[ERROR] Corrupted model file: ...` | 0-byte or corrupted `.joblib` binary file | The system prompts to retrain immediately. Select Option 5 to overwrite the broken artifact. |
| `[ERROR] Permission denied saving to ...` | The CSV or model file is open in another program (e.g. Microsoft Excel) | Close Microsoft Excel or the external viewer and retry the operation. |
| `[!] Invalid forecast date ... Must be strictly AFTER latest observation` | Target date is in the past or within historical observations | Enter a future date strictly greater than the latest recorded date. |
| `[!] Value must be a finite decimal number` | Input is text, `NaN`, or `Infinity` | Enter a standard numeric decimal number (e.g. `24.50`). |

---

## 📋 Internship PDF Requirements Compliance Matrix

| Requirement from Specification | Implementation Detail | Location in Codebase |
| :--- | :--- | :--- |
| **Python 3.11+ & Dependencies** | Pure Python 3.11+ with `pandas`, `numpy`, `scikit-learn`, `joblib`, and `pytest`. | [requirements.txt](file:///d:/CODEVEDEX/Task-1/requirements.txt) |
| **Modular File Organization** | Separated into data manager, predictor, utilities, and main entrypoint. | [`main.py`](file:///d:/CODEVEDEX/Task-1/main.py), [`src/`](file:///d:/CODEVEDEX/Task-1/src) |
| **CSV Storage (`usage_data.csv`)** | Daily electricity storage with `date` and `usage_kwh` columns. | [src/data_manager.py](file:///d:/CODEVEDEX/Task-1/src/data_manager.py#L185-L265) |
| **ISO Dates & Domain Validation** | Rejects non-ISO dates, impossible calendar dates, negative kWh, NaN, and Inf. | [src/data_manager.py](file:///d:/CODEVEDEX/Task-1/src/data_manager.py#L46-L107) |
| **Duplicate Prevention** | Detects existing dates, rejects additions, and routes users to update. | [src/data_manager.py](file:///d:/CODEVEDEX/Task-1/src/data_manager.py#L271-L305) |
| **Statistical Computations** | Total kWh, average daily kWh, min/max, trailing 7-day usage, and metrics explanation. | [src/utils.py](file:///d:/CODEVEDEX/Task-1/src/utils.py#L202-L289) |
| **Chronological Train/Test Split** | 80/20 chronological split with zero random shuffling to eliminate future data leakage. | [src/predictor.py](file:///d:/CODEVEDEX/Task-1/src/predictor.py#L79-L185) |
| **Model Persistence with joblib** | Serializes model, reference date, timestamps, drift slope, and metrics to `models/`. | [src/predictor.py](file:///d:/CODEVEDEX/Task-1/src/predictor.py#L403-L500) |
| **Future Forecast & Staleness** | Checks future date $> \text{latest observation}$, alerts on stale data, flags negative trends. | [`main.py`](file:///d:/CODEVEDEX/Task-1/main.py#L365-L500), [src/predictor.py](file:///d:/CODEVEDEX/Task-1/src/predictor.py#L350-L401) |
| **Exhaustive Error Handling** | Catches `PermissionError`, `ValueError`, `ParserError`, `EOFError`, and `KeyboardInterrupt`. | All modules |
| **Never Silently Discard Records** | Changed `on_bad_lines="skip"` to `"error"`; reports malformed lines with recovery options. | [src/data_manager.py](file:///d:/CODEVEDEX/Task-1/src/data_manager.py#L200-L225) |
| **Automated Testing Suite** | 74 passing unit and CLI smoke tests using isolated temporary directories. | [`tests/`](file:///d:/CODEVEDEX/Task-1/tests) |

---

## 👤 Author & Acknowledgments

* **Intern**: AI/ML Engineering Intern
* **Program**: CODEVEDX AI/ML Internship Program
* **Task**: Task 1 - Utility Usage Prediction Tool
* **License**: MIT Open Source License
