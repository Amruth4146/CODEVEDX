# CODEVEDX AI/ML Internship Portfolio

[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Verification: 100% PASS](https://img.shields.io/badge/Audit-100%25%20PASS-brightgreen.svg)]()

Welcome to my official **CODEVEDX Artificial Intelligence & Machine Learning (AI/ML) Internship** repository. This repository hosts all assigned tasks, project implementations, automated test suites, and formal submission deliverables.

---

## 📂 Repository Structure

`	ext
CODEVEDX/
│
├── README.md                                  # Repository overview and task directory
├── .gitignore                                 # Top-level version control ignore rules
│
├── Task-1/                                    # Task 1: Utility Usage Prediction Tool
│   ├── main.py                                # Interactive CLI application dispatcher
│   ├── requirements.txt                       # Production & testing dependencies
│   ├── README.md                              # Detailed task documentation & quickstart
│   ├── .gitignore                             # Task-specific ignore patterns
│   │
│   ├── src/                                   # Core business logic & ML modules
│   │   ├── __init__.py
│   │   ├── data_manager.py                    # CSV I/O, validation, feature extraction
│   │   ├── predictor.py                       # OLS LinearRegression baseline & inference
│   │   └── utils.py                           # NumPy/Pandas math, formatting, prompt helpers
│   │
│   ├── data/                                  # Historical datasets
│   │   └── usage_data.csv                     # 732 daily electricity consumption records
│   │
│   ├── models/                                # Serialized machine learning artifacts
│   │   └── electricity_model.joblib           # Trained model with metadata & hash
│   │
│   ├── tests/                                 # Automated pytest verification suite
│   │   ├── test_cli_smoke.py                  # End-to-end CLI workflow tests
│   │   ├── test_data_manager.py               # Data I/O, calendar, and boundary tests
│   │   ├── test_main.py                       # Interactive loop and interrupt tests
│   │   ├── test_predictor.py                  # ML training, split, and metric tests
│   │   └── test_utils.py                      # Statistics and prompt helper tests
│   │
│   └── docs/                                  # Official submission documentation
│       ├── CODEVEDX_Task1_Project_Report.pdf  # 12-page executive project report (PDF)
│       ├── CODEVEDX_Task1_Presentation.pptx   # 12-slide widescreen presentation (PPTX)
│       ├── ARCHITECTURE.md                    # Technical architecture specification
│       ├── USAGE_GUIDE.md                     # PowerShell execution instructions
│       ├── figures/                           # High-resolution architectural diagrams
│       └── screenshots/                       # Visual terminal interface captures
│
└── Task-2/                                    # Task 2: Student Performance Prediction System (EduPredict AI)
    ├── app.py                                 # Main Streamlit web application & landing portal
    ├── requirements.txt                       # Production cloud runtime dependencies
    ├── requirements-dev.txt                   # Local development & test suite dependencies
    ├── README.md                              # Comprehensive 20-section project documentation
    ├── CONTRIBUTING.md                        # Enterprise GitFlow contribution guidelines
    ├── .gitignore                             # Task-specific ignore patterns
    │
    ├── .streamlit/
    │   └── config.toml                        # Cloud server settings & Light SaaS theme
    │
    ├── assets/
    │   └── styles.css                         # SaaS design system, elevation tokens & media queries
    │
    ├── data/                                  # Academic cohort datasets
    │   ├── raw/
    │   │   └── student_data.csv               # 750-sample primary academic dataset
    │   └── processed/
    │       └── cleaned_student_data.csv       # Preprocessed, clamped, and sanitized dataset
    │
    ├── models/                                # Serialized Scikit-learn Pipeline artifacts
    │   └── student_performance_model.joblib   # Persisted champion model bundle
    │
    ├── notebooks/
    │   └── eda.ipynb                          # Jupyter notebook for exploratory data analysis
    │
    ├── pages/                                 # Streamlit multi-page application modules
    │   ├── 1_Dashboard.py                     # Executive Dashboard & cohort telemetry
    │   ├── 2_Data_Analysis.py                 # Multi-dimensional EDA, filters, and heatmaps
    │   ├── 3_Performance_Predictor.py         # Single/batch inference & What-If simulator
    │   ├── 4_Model_Evaluation.py              # Benchmark leaderboard, parity plots & drift checks
    │   ├── 5_Dataset_Manager.py               # Catalog browser, health inspector & CSV upload
    │   └── 6_About_Project.py                 # Architecture flowcharts & internship rubric
    │
    ├── src/                                   # Core architecture & business logic modules
    │   ├── __init__.py
    │   ├── data_loader.py                     # Dataset loading, caching, benchmark generator
    │   ├── preprocessing.py                   # Imputation, range clamping, schema aliasing
    │   ├── eda.py                             # Interactive Plotly charts & statistical summaries
    │   ├── feature_selection.py               # Pearson, Spearman, MI, and Random Forest MDI
    │   ├── train_model.py                     # KFold CV, GridSearchCV tuning, serialization
    │   ├── evaluation.py                      # Regression metrics, parity & residual diagnostics
    │   ├── prediction.py                      # ModelPredictor class for single & batch inference
    │   ├── security.py                        # PII redaction, file limits, deserialization checks
    │   └── ui_components.py                   # Reusable headers, KPI cards, sidebar, themes
    │
    ├── tests/                                 # Automated pytest verification suite (72 tests)
    │   ├── __init__.py
    │   ├── test_eda.py                        # EDA chart rendering & filter tests (9 tests)
    │   ├── test_evaluation.py                 # Metric calculations & residual tests (12 tests)
    │   ├── test_feature_selection.py          # Correlation, MI, MDI, and nested CV (8 tests)
    │   ├── test_prediction.py                 # Single/batch inference & simulator (6 tests)
    │   ├── test_preprocessing.py              # Imputation, range clamping & encoding (13 tests)
    │   ├── test_qa_audit.py                   # Security, PII, extreme input & robust QA (16 tests)
    │   └── test_train_model.py                # Train-test split, CV selection & schema (8 tests)
    │
    └── docs/                                  # Verified screenshots & presentation guides
        ├── DEMO_GUIDE.md                      # 3–5 min presentation script, storyboard & Q&A
        └── screenshots/                       # 8 high-resolution application screenshots
`

---

## 📌 Internship Tasks Overview

### [Task 1: Utility Usage Prediction Tool](./Task-1/README.md)
* **Domain:** Predictive Machine Learning & Data Systems
* **Objective:** Design and build an interactive, fail-safe console application in Python 3.11+ to manage daily electricity consumption records (kWh), calculate descriptive statistics, fit an Ordinary Least Squares (OLS) linear trend baseline using strict chronological splitting (no data leakage), and forecast future energy demand.
* **Key Achievements:**
  - **74 of 74 Automated Tests Passed** (100% pass rate in 3.90s).
  - Comprehensive data validation: ISO-8601 calendar validity, non-negative kWh limits, duplicate date rejection.
  - Zero future-data leakage: Evaluated strictly on forward chronological test split (80% train / 20% test).
  - Production-grade crash immunity: Intercepts KeyboardInterrupt (Ctrl+C), EOF signals, and invalid inputs cleanly without unhandled stack traces.
* **Deliverables:**
  - 📄 **Project Report:** Task-1/docs/CODEVEDX_Task1_Project_Report.pdf
  - 📊 **Presentation Deck:** Task-1/docs/CODEVEDX_Task1_Presentation.pptx

---

### [Task 2: Student Performance Prediction System (EduPredict AI)](./Task-2/README.md)
* **Domain:** Artificial Intelligence & Machine Learning / Web Applications
* **Objective:** Build an enterprise-grade, multi-page analytics platform and predictive machine learning tool in Python 3.11 and Streamlit to analyze academic cohort telemetry, explore educational trends with interactive Plotly visualizations, benchmark regression algorithms with 5-fold cross-validation, and forecast student examination performance with real-time What-If sensitivity simulations.
* **Key Achievements:**
  - **72 of 72 Automated Tests Passed** (100% pass rate across unit, integration, and security suites).
  - **Superior Generalization:** Selected OLS Linear Regression champion via 5-Fold Cross-Validation (CV RMSE: **3.15 marks** vs. Random Forest: **4.03 marks**).
  - **Unbiased Test Performance:** Explains **86.41% of variance** (^2 = 0.8641$) with an average test error of only $\pm 2.78$ marks (Test RMSE: **3.42 marks**, MAE: **2.78 marks**).
  - **Zero Data Leakage:** Preprocessing (median imputation, feature scaling) encapsulated strictly inside Scikit-Learn Pipelines fitted on training folds.
  - **Interactive Decision-Support:** What-If Study Simulator (+1h, +2h, +3h sensitivity modeling), academic category stratification (*Distinction*, *First Class*, *Pass*, *At-Risk*), and batch CSV scoring engine.
  - **Enterprise Security & Privacy:** Integrated 15MB file limits, regex-based student PII detection/redaction, safe Joblib deserialization validation, and FERPA-compliant privacy safeguards.
* **Deliverables:**
  - 🌐 **Web Application:** Task-2/app.py (Multi-page Streamlit Cloud ready)
  - 📘 **Technical Documentation:** Task-2/README.md (Comprehensive 20-section specification)
  - 🎙️ **Demonstration Guide:** Task-2/docs/DEMO_GUIDE.md (3–5 minute presentation script, storyboard & Q&A)
  - 🖼️ **Visual Artifacts:** Task-2/docs/screenshots/ (8 verified high-resolution application screenshots)

---

## 🛠️ Global Setup & Prerequisites

Each task is self-contained within its own directory with its own dependencies and test suites:

### Task 1: Utility Usage Prediction Tool (CLI)
`powershell
# Clone the repository
git clone https://github.com/amruth4146/CODEVEDX.git
cd CODEVEDX\Task-1

# Install dependencies
py -m pip install -r requirements.txt

# Run the test suite (74 tests)
py -m pytest -v

# Launch the interactive CLI application
py main.py
`

### Task 2: Student Performance Prediction System (Web App)
`powershell
# Navigate to Task-2
cd CODEVEDX\Task-2

# Install production dependencies (or requirements-dev.txt for tests)
py -m pip install -r requirements.txt

# Run the automated test suite (72 tests)
py -m pytest -v

# Launch the interactive Streamlit web application
py -m streamlit run app.py
`

---

## 👨‍💻 Author & Intern Information
* **Candidate:** amruth4146
* **Program:** CODEVEDX AI/ML Internship
* **Submission Date:** October 2026
