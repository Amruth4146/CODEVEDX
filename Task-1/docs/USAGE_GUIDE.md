# User Guide & PowerShell Execution Instructions

This guide provides step-by-step instructions for running the **Electricity Consumption Predictor** on **Windows PowerShell**.

---

## 1. Prerequisites
- **Python 3.11+** installed on Windows.
- Windows PowerShell terminal.

---

## 2. Environment Setup (PowerShell)

### Step 2.1: Open PowerShell in the Project Directory
```powershell
cd d:\CODEVEDEX\Task-1
```

### Step 2.2: Create and Activate Virtual Environment
```powershell
# Create Python 3.11 virtual environment
py -3.11 -m venv .venv

# Set execution policy for the current session (if script execution is restricted)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 2.3: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 3. Running the Application

### 3.1 Interactive Console Menu (Default)
To launch the user-friendly interactive menu, simply run:
```powershell
python main.py
```
This presents the following options:
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

---

## 4. Command-Line Options (Non-Interactive)

You can also run all tasks directly from PowerShell without entering the interactive menu:

### 4.1 Generate / Reset Demonstration Dataset
```powershell
python main.py --generate-data
```

### 4.2 Inspect Dataset Summary Statistics
```powershell
python main.py --summary
```

### 4.3 View Historical Records
```powershell
# View latest 10 records
python main.py --view-records 10
```

### 4.4 Add a New Consumption Record
```powershell
python main.py --add-record 2025-01-01 24.80
```

### 4.5 Update an Existing Record by Date
```powershell
python main.py --update-record 2025-01-01 26.10
```

### 4.6 Train & Evaluate the ML Model
```powershell
python main.py --train
```

### 4.7 View Trained Model Information & Metrics
```powershell
python main.py --model-info
```

### 4.8 Predict Single Day Consumption (CLI)
```powershell
python main.py --predict --date 2025-07-15 --lag1 26.5 --lag7 25.0 --rolling 25.8
```

### 4.9 Batch Forecast on Recent Test Split
```powershell
python main.py --batch-predict 7
```

---

## 5. Running Automated Unit Tests

Run the complete test suite with verbose output:
```powershell
pytest -v
```

All 30 unit tests validate:
- ISO date formats (e.g. leap years, format errors)
- Finite non-negative usage values (rejecting NaN, Inf, negative kWh)
- Duplicate date handling and file safety flags
- Model training, evaluation, saving, and loading
- Console prompt and CLI utilities
