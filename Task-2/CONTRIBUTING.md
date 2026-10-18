# 🤝 Contributing to Student Performance Prediction System

Thank you for contributing to the **Student Performance Prediction System** (CODEVEDX Task 2)!  
This document outlines our Git branching standards, code formatting expectations, and pull request workflow to maintain high software engineering and data science rigor.

---

## 🌿 Branching Strategy

We follow a modified **GitFlow** architecture. All development must occur on dedicated feature branches branched from `develop`.

| Branch | Purpose | Base Branch | Target Merge |
| :--- | :--- | :--- | :--- |
| **`main`** | Production-ready, stable releases. Fully tested and tagged. | N/A | Production Deployments |
| **`develop`** | Active integration branch. Accumulates tested features. | `main` | `main` (for releases) |
| **`feature/*`** | Specific milestone features (e.g., `feature/eda`, `feature/model-training`). | `develop` | `develop` |
| **`fix/*`** | Bug fixes or issue patches. | `develop` | `develop` |
| **`docs/*`** | Documentation improvements. | `develop` | `develop` |

---

## 🚀 Development & Contribution Workflow

### 1. Synchronize and Create a Feature Branch
Always branch off the latest `develop` branch:
```bash
git checkout develop
git pull origin develop
git checkout -b feature/<feature-name>
```

### 2. Follow Conventional Commits
Write concise, descriptive commit messages using the **Conventional Commits** specification:

- `feat:` A new feature or capability (e.g., `feat(eda): add interactive correlation heatmap`)
- `fix:` A bug fix (e.g., `fix(preprocessing): resolve boundary clamping for negative study hours`)
- `docs:` Documentation updates only (e.g., `docs: add Git branching guide to README`)
- `style:` Formatting or UI styling changes (e.g., `style(ui): update KPI card elevations and padding`)
- `refactor:` Code restructuring without changing functional behavior
- `test:` Adding or updating unit tests (e.g., `test(ml): add cross-validation leakage checks`)
- `chore:` Dependency or build system updates (e.g., `chore: update requirements.txt`)

### 3. Run Automated Tests Locally
Before opening a Pull Request, all 56 automated tests must pass without errors:
```bash
# Execute full pytest suite
python -m pytest tests/ -v
```

### 4. Open a Pull Request (PR)
- Push your feature branch:
  ```bash
  git push origin feature/<feature-name>
  ```
- Open a Pull Request targeting **`develop`** (never directly target `main`).
- Fill in the PR description detailing:
  1. What changes were made.
  2. Testing performed and evidence (pytest output).
  3. UI screenshots if frontend changes were introduced.

### 5. Merge Strategy
- Merge PRs into `develop` using non-fast-forward merge (`--no-ff`) to preserve branch history:
  ```bash
  git checkout develop
  git merge --no-ff feature/<feature-name>
  ```
- When preparing a production milestone release, merge `develop` into `main` and tag the release:
  ```bash
  git checkout main
  git merge --no-ff develop
  git tag -a v2.4.0 -m "Release v2.4.0: Production-ready EduPredict AI platform"
  ```

---

## 🛡️ Code & Machine Learning Guidelines

1. **No Data Leakage**: Transformations (scalers, imputers) must be fit on training splits only and applied to test/validation sets.
2. **Deterministic Modeling**: Always fix `random_state=42` in data splits, cross-validation folds, and stochastic estimators.
3. **No Target Imputation**: Never impute missing target scores (`final_exam_marks`). Rows with missing targets must be excluded from supervised training.
4. **Accessible UI**: All UI components must comply with responsive layout standards and maintain adequate color contrast.
5. **Clean Secrets**: Never commit API keys, personal credentials, or `.env` files.
