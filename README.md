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
└── Task-1/                                    # Task 1: Utility Usage Prediction Tool
    ├── main.py                                # Interactive CLI application dispatcher
    ├── requirements.txt                       # Production & testing dependencies
    ├── README.md                              # Detailed task documentation & quickstart
    ├── .gitignore                             # Task-specific ignore patterns
    │
    ├── src/                                   # Core business logic & ML modules
    │   ├── __init__.py
    │   ├── data_manager.py                    # CSV I/O, validation, feature extraction
    │   ├── predictor.py                       # OLS LinearRegression baseline & inference
    │   └── utils.py                           # NumPy/Pandas math, formatting, prompt helpers
    │
    ├── data/                                  # Historical datasets
    │   └── usage_data.csv                     # 732 daily electricity consumption records
    │
    ├── models/                                # Serialized machine learning artifacts
    │   └── electricity_model.joblib           # Trained model with metadata & hash
    │
    ├── tests/                                 # Automated pytest verification suite
    │   ├── test_cli_smoke.py                  # End-to-end CLI workflow tests
    │   ├── test_data_manager.py               # Data I/O, calendar, and boundary tests
    │   ├── test_main.py                       # Interactive loop and interrupt tests
    │   ├── test_predictor.py                  # ML training, split, and metric tests
    │   └── test_utils.py                      # Statistics and prompt helper tests
    │
    └── docs/                                  # Official submission documentation
        ├── CODEVEDX_Task1_Project_Report.pdf  # 12-page executive project report (PDF)
        ├── CODEVEDX_Task1_Presentation.pptx   # 12-slide widescreen presentation (PPTX)
        ├── ARCHITECTURE.md                    # Technical architecture specification
        ├── USAGE_GUIDE.md                     # PowerShell execution instructions
        ├── figures/                           # High-resolution architectural diagrams
        └── screenshots/                       # Visual terminal interface captures
`

---

## 🚀 Internship Tasks Overview

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

## 🛠️ Global Setup & Prerequisites

Each task is self-contained within its own directory. To set up the environment:

`powershell
# Clone the repository
git clone https://github.com/amruth4146/CODEVEDX.git
cd CODEVEDX\Task-1

# Install dependencies
py -m pip install -r requirements.txt

# Run the test suite
py -m pytest -v

# Launch the interactive application
py main.py
`

---

## 👤 Author & Intern Information
* **Candidate:** amruth4146
* **Program:** CODEVEDX AI/ML Internship
* **Submission Date:** October 2026
