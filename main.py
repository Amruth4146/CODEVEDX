"""Main application entrypoint: CODEVEDX | Utility Usage Prediction Tool.

This module provides a pure console-based, professional interactive CLI.
It connects user interactions with the underlying business logic in data_manager
and predictor modules while ensuring safe input handling, graceful exception
management (preventing ugly tracebacks on Ctrl+C, EOF, and invalid choices),
and aligned formatting.
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

# Ensure UTF-8 output if supported to avoid Windows PowerShell codepage encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd

from src.data_manager import (
    SYNTHETIC_DATA_DISCLAIMER,
    add_record,
    extract_features,
    generate_synthetic_sample_data,
    get_dataset_summary,
    load_data,
    preprocess_data,
    save_data,
    split_data,
    update_record,
    validate_data,
    validate_iso_date,
    validate_usage_value,
    view_all_records,
)
from src.predictor import ConsumptionPredictor
from src.utils import (
    DATE_COLUMN,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    TARGET_COLUMN,
    confirm_action,
    display_usage_statistics,
    format_kwh,
    print_header,
    print_metric,
    print_section,
    prompt_date,
    prompt_float,
)

APPLICATION_TITLE = "CODEVEDX | Utility Usage Prediction Tool"


# =====================================================================
# Safe Input & UI Helpers
# =====================================================================

def safe_input(prompt_str: str = "") -> Optional[str]:
    """Safely prompt the user for input without raising ugly tracebacks on Ctrl+C or EOF.

    Args:
        prompt_str: Prompt string to display.

    Returns:
        Stripped string input, or None if user interrupted (Ctrl+C) or triggered EOF.
    """
    try:
        return input(prompt_str).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return None


def pause_for_user() -> None:
    """Prompt the user to press Enter before returning to the menu, safely handling interrupts."""
    try:
        input("\nPress [Enter] to return to the main menu...")
    except (KeyboardInterrupt, EOFError):
        print()


def display_menu() -> None:
    """Display the application title and official numbered menu."""
    line = "=" * 65
    print(f"\n{line}")
    print(f"  {APPLICATION_TITLE}")
    print(f"{line}")
    print("  1. View usage records")
    print("  2. Add usage record")
    print("  3. Update usage record")
    print("  4. View usage statistics")
    print("  5. Train ML model")
    print("  6. Predict future usage")
    print("  7. View model evaluation")
    print("  8. Exit")
    print("-" * 65)


def ensure_dataset_exists(data_path: Path = DEFAULT_DATA_PATH) -> Optional[pd.DataFrame]:
    """Check if the dataset exists, offering to generate sample data if missing.

    Args:
        data_path: Path to usage CSV.

    Returns:
        Loaded DataFrame, or None if user declines generation.
    """
    if not data_path.is_file():
        print(f"\n[WARNING] Dataset not found at: '{data_path}'.")
        create_now = confirm_action(
            "Would you like to generate demonstration sample records?", default=True
        )
        if not create_now:
            print("[INFO] Operation aborted. A dataset is required.")
            return None

        print("\n[+] Generating demonstration synthetic dataset (730 daily records)...")
        print(f"    DISCLAIMER: {SYNTHETIC_DATA_DISCLAIMER}")
        df = generate_synthetic_sample_data(output_path=data_path, allow_overwrite=True)
        print(f"[SUCCESS] Sample dataset created with {len(df)} records at: {data_path}")
        return df

    try:
        return load_data(data_path)
    except Exception as exc:
        print(f"\n[ERROR] Failed to load dataset: {exc}")
        return None


# =====================================================================
# Menu Action Handlers
# =====================================================================

def menu_view_records(data_path: Path = DEFAULT_DATA_PATH) -> None:
    """Option 1: View usage records."""
    print_header("1. View Usage Records")
    df = ensure_dataset_exists(data_path)
    if df is None:
        return

    total_count = len(df)
    print(f"Total recorded days: {total_count}")
    print("Display options:")
    print("  [1] View latest 20 records (default)")
    print("  [2] View all records")
    print("  [3] Custom number of recent records")

    sub_choice = safe_input("Select an option [1-3, default 1]: ")
    if sub_choice is None:
        print("[INFO] Operation cancelled.")
        return

    if sub_choice == "2":
        limit = None
        records = view_all_records(df)
        print(f"\nDisplaying all {len(records)} records:")
    elif sub_choice == "3":
        custom_limit = prompt_float("Enter number of recent records to view", min_val=1, default=20)
        limit = int(custom_limit)
        records = view_all_records(df, limit=limit, tail=True)
        print(f"\nDisplaying latest {len(records)} records:")
    else:
        limit = 20
        records = view_all_records(df, limit=limit, tail=True)
        print(f"\nDisplaying latest {len(records)} records:")

    print("\n  " + f"{'Index':<6} | {'Date (YYYY-MM-DD)':<18} | {'Daily Usage (kWh)':<18}")
    print("  " + "-" * 50)
    for idx, row in records.iterrows():
        dt_str = str(row["date"])[:10]
        usage = float(row[TARGET_COLUMN])
        print(f"  {idx + 1:<6} | {dt_str:<18} | {usage:<18.2f}")


def menu_add_record(data_path: Path = DEFAULT_DATA_PATH) -> None:
    """Option 2: Add usage record."""
    print_header("2. Add Usage Record")
    df = ensure_dataset_exists(data_path)
    if df is None:
        return

    print("Enter the details of the record to add (or press Ctrl+C to cancel):")
    try:
        date_input = prompt_date("  Date (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))
        date_str = date_input.strftime("%Y-%m-%d")
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Operation cancelled.")
        return

    # Check for existing duplicate date
    dates_in_dataset = df["date"].astype(str).str[:10].values
    if date_str in dates_in_dataset:
        print(f"\n[WARNING] A record for date '{date_str}' already exists.")
        print("          Please use Menu Option 3 ('Update usage record') to modify it.")
        return

    try:
        usage_val = prompt_float("  Electricity Usage (kWh)", min_val=0.0)
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Operation cancelled.")
        return

    print(f"\nSummary of new record:")
    print(f"  * Date : {date_str}")
    print(f"  * Usage: {usage_val:.2f} kWh")

    proceed = confirm_action("Do you want to save this record?", default=True)
    if not proceed:
        print("[INFO] Record addition cancelled. No changes were saved.")
        return

    try:
        updated_df = add_record(df, date_str=date_str, usage_kwh=usage_val)
        save_data(updated_df, file_path=data_path, allow_overwrite=True)
        print(f"\n[SUCCESS] Successfully added record for {date_str} ({usage_val:.2f} kWh).")
        print(f"          Total records in dataset: {len(updated_df)}.")
    except Exception as exc:
        print(f"\n[ERROR] Failed to save record: {exc}")


def menu_update_record(data_path: Path = DEFAULT_DATA_PATH) -> None:
    """Option 3: Update usage record."""
    print_header("3. Update Usage Record")
    df = ensure_dataset_exists(data_path)
    if df is None:
        return

    print("Enter the date of the record you wish to update:")
    try:
        date_input = prompt_date("  Target Date (YYYY-MM-DD)")
        date_str = date_input.strftime("%Y-%m-%d")
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Operation cancelled.")
        return

    # Locate existing record
    dates_str = df["date"].astype(str).str[:10]
    match = df[dates_str == date_str]

    if match.empty:
        print(f"\n[ERROR] No record found for date '{date_str}'.")
        print("        Use Menu Option 2 ('Add usage record') if you want to add a new record.")
        return

    current_val = float(match.iloc[0][TARGET_COLUMN])
    print(f"\nFound existing record for {date_str}:")
    print(f"  * Current Electricity Usage: {current_val:.2f} kWh")

    try:
        new_val = prompt_float("  Enter New Electricity Usage (kWh)", min_val=0.0, default=current_val)
    except (KeyboardInterrupt, EOFError):
        print("\n[INFO] Operation cancelled.")
        return

    if abs(new_val - current_val) < 1e-4:
        print("\n[INFO] New value is identical to current value. No update needed.")
        return

    proceed = confirm_action(
        f"Confirm update for {date_str} from {current_val:.2f} kWh to {new_val:.2f} kWh?",
        default=True,
    )
    if not proceed:
        print("[INFO] Record update cancelled. No changes were saved.")
        return

    try:
        updated_df = update_record(df, date_str=date_str, new_usage_kwh=new_val)
        save_data(updated_df, file_path=data_path, allow_overwrite=True)
        print(f"\n[SUCCESS] Record for {date_str} successfully updated to {new_val:.2f} kWh.")
    except Exception as exc:
        print(f"\n[ERROR] Failed to update record: {exc}")


def menu_view_statistics(data_path: Path = DEFAULT_DATA_PATH) -> None:
    """Option 4: View usage statistics."""
    print_header("4. View Usage Statistics")
    df = ensure_dataset_exists(data_path)
    if df is None:
        return

    display_usage_statistics(df)


def menu_train_model(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    test_size: float = 0.2,
) -> None:
    """Option 5: Train ML model."""
    print_header("5. Train Machine Learning Model")
    df = ensure_dataset_exists(data_path)
    if df is None:
        return

    if len(df) < 5:
        print(f"\n[WARNING] Dataset contains only {len(df)} records. Minimum 5 records required.")
        generate_more = confirm_action(
            "Would you like to generate demonstration sample records?", default=True
        )
        if not generate_more:
            print("[INFO] Training aborted.")
            return
        df = generate_synthetic_sample_data(days=365, output_path=data_path, allow_overwrite=True)

    if model_path.is_file():
        retrain = confirm_action(
            "A trained model already exists. Do you want to retrain and overwrite it?",
            default=True,
        )
        if not retrain:
            print("[INFO] Training cancelled. Existing model retained.")
            return

    print("\nStarting chronological model training pipeline...")
    print("NOTE: This model is a simple trend-based baseline (day_index -> usage_kwh),")
    print("      not an accurate seasonal forecasting system. Never invent model accuracy.\n")

    try:
        print("[1/3] Deriving numeric day index from chronological dates (preventing data leakage)...")
        print(f"[2/3] Chronologically splitting observations (Train: {int((1-test_size)*100)}%, Test: {int(test_size*100)}%)...")
        predictor = ConsumptionPredictor()
        metrics = predictor.train_chronological(df, test_size=test_size)

        print("[3/3] Fitted LinearRegression baseline and evaluated on forward test observations.")

        print_section("Out-of-Sample Evaluation Metrics")
        print_metric("Mean Absolute Error (MAE)", metrics["mae"], unit="kWh", decimal_places=4)
        print_metric("Root Mean Squared Error (RMSE)", metrics["rmse"], unit="kWh", decimal_places=4)
        if metrics.get("r2_score") is not None:
            print_metric("R^2 Score (Variance Explained)", metrics["r2_score"], decimal_places=4)
        else:
            print("  * R^2 Score (Variance Explained) : Undefined (insufficient test variance)")

        meta = predictor.training_metadata
        if meta:
            print_section("Baseline Model Formula")
            slope = meta.get("slope_kwh_per_day", 0.0)
            intercept = meta.get("intercept_kwh", 0.0)
            sign = "+" if slope >= 0 else "-"
            print(f"  * Linear Formula              : usage_kwh = {intercept:.4f} {sign} {abs(slope):.4f} * day_index")
            print(f"  * Reference Base Date (Day 0) : {meta.get('reference_date', 'N/A')}")
            print(f"  * Daily Trend Rate (Slope)    : {slope:+.4f} kWh/day")
            print(f"  * Training Observations       : {meta.get('train_records', 'N/A')} records ({meta.get('train_date_range', 'N/A')})")
            print(f"  * Testing Observations        : {meta.get('test_records', 'N/A')} records ({meta.get('test_date_range', 'N/A')})")

        saved_file = predictor.save(model_path)
        print(f"\n[SUCCESS] Model successfully trained and serialized to: '{saved_file}'.")
    except Exception as exc:
        print(f"\n[ERROR] Model training failed: {exc}")


def menu_predict_future(
    data_path: Path = DEFAULT_DATA_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    """Option 6: Predict future electricity consumption for a date after the latest historical observation."""
    print_header("6. Predict Future Usage")

    # Step 1: Verify model artifact exists; if missing, explain how to train it
    if not model_path.is_file():
        print(f"[WARNING] Trained model artifact not found at: '{model_path}'.")
        print("\nTo train a prediction model:")
        print("  1. Return to the main menu.")
        print("  2. Select Option 5 ('Train ML model').")
        print("  3. The system will fit a LinearRegression trend baseline on your historical data.")
        print("  4. The model and its chronological metadata will be serialized to models/.\n")

        train_now = confirm_action("Would you like to run the training pipeline now?", default=True)
        if not train_now:
            print("[INFO] Prediction aborted. A trained model is required.")
            return
        menu_train_model(data_path=data_path, model_path=model_path)
        if not model_path.is_file():
            print("[ERROR] Training was not completed. Cannot proceed with prediction.")
            return

    try:
        predictor = ConsumptionPredictor.load(model_path)
    except Exception as exc:
        print(f"\n[ERROR] Failed to load model artifact: {exc}")
        return

    # Step 2: Ensure historical dataset exists for date boundary checking
    df = ensure_dataset_exists(data_path)
    if df is None or df.empty:
        print("[ERROR] Cannot validate forecast date: historical dataset is empty.")
        return

    # Step 3: Detect stale model metadata if data changed after training
    is_stale, stale_reason = predictor.check_model_staleness(df)
    if is_stale:
        print_section("Model Staleness Detected")
        print(f"  [WARNING] {stale_reason}")
        print("  Predictions made with an outdated model may not reflect recent consumption patterns.")
        retrain_now = confirm_action(
            "Would you like to retrain the model now with the updated dataset?",
            default=True,
        )
        if retrain_now:
            menu_train_model(data_path=data_path, model_path=model_path)
            try:
                predictor = ConsumptionPredictor.load(model_path)
            except Exception as exc:
                print(f"[ERROR] Failed to reload retrained model: {exc}")
                return
        else:
            print("  [WARNING] Continuing with stale model as requested.\n")

    # Step 4: Determine latest historical observation
    clean_dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce").dropna()
    if clean_dates.empty:
        print("[ERROR] Historical dataset contains no valid calendar dates.")
        return

    latest_historical_dt = clean_dates.max()
    latest_historical_str = latest_historical_dt.strftime("%Y-%m-%d")

    print(f"Latest historical observation recorded: {latest_historical_str}")
    print("Please enter a future forecast date (must be strictly AFTER the latest observation):")

    # Step 5: Prompt and validate that date is strictly after latest observation
    while True:
        try:
            target_dt = prompt_date("  Future Forecast Date")
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Operation cancelled.")
            return

        if target_dt.date() <= latest_historical_dt.date():
            print(
                f"  [!] Invalid forecast date '{target_dt.strftime('%Y-%m-%d')}'. "
                f"Must be strictly AFTER the latest historical observation ({latest_historical_str})."
            )
            continue
        break

    # Step 6: Use identical date-to-numeric day index transformation
    ref_date_str = (
        predictor.reference_date.strftime("%Y-%m-%d")
        if predictor.reference_date
        else "Dataset start"
    )
    if predictor.reference_date:
        calc_day_idx = (target_dt.date() - predictor.reference_date.date()).days
    else:
        calc_day_idx = 0

    print(f"\n  * Reference Base Date (Day 0): {ref_date_str}")
    print(f"  * Numeric Day Index          : Day {calc_day_idx}")

    proceed = confirm_action(
        f"Proceed with forecast for {target_dt.strftime('%Y-%m-%d')} (Day {calc_day_idx})?",
        default=True,
    )
    if not proceed:
        print("[INFO] Prediction cancelled.")
        return

    try:
        predicted_kwh = predictor.predict_for_date(date_input=target_dt)
        day_name = target_dt.strftime("%A")
        is_wknd = "Weekend" if target_dt.weekday() >= 5 else "Weekday"

        print_section("Future Electricity Usage Forecast")
        print(f"  Forecast Date                : {target_dt.strftime('%Y-%m-%d')} ({day_name}, {is_wknd})")
        print(f"  Reference Base Date (Day 0)  : {ref_date_str}")
        print(f"  Elapsed Numeric Day Index    : {calc_day_idx} days")

        # Step 7: Flag negative predictions as unsuitable model results
        if predicted_kwh < 0.0:
            print("  " + "=" * 55)
            print(f"  * RAW PREDICTED CONSUMPTION  : {predicted_kwh:.2f} kWh")
            print("  " + "=" * 55)
            print("\n  [WARNING] UNSUITABLE MODEL RESULT: Negative Consumption Forecast!")
            print(f"  * The linear model projected a negative usage value of {predicted_kwh:.2f} kWh.")
            print("  * Physical electricity consumption cannot be negative (>= 0.0 kWh).")
            print("  * Cause: Extrapolating a downward trend line past its zero-intercept point.")
            print("  * This indicates that a simple linear trend is NOT suitable for this forecast horizon.")
            print("  * Do NOT treat this figure as a reliable or usable forecast.")
        else:
            print("  " + "=" * 55)
            print(f"  * PREDICTED DAILY CONSUMPTION: {predicted_kwh:.2f} kWh")
            print("  " + "=" * 55)

        # Step 8: Uncertainty and data quality advisory
        print("\n  [NOTE] Forecast Uncertainty Advisory:")
        print("  * Future estimates carry inherent uncertainty and heavily depend on historical")
        print("    training data quality, sample duration, and linear extrapolation assumptions.")
        print("  * Linear trend models do not capture weather fluctuations, cooling/heating peaks,")
        print("    or sudden demand shifts. Use as a baseline estimate only.")

    except Exception as exc:
        print(f"\n[ERROR] Prediction failed: {exc}")


def menu_view_model_evaluation(model_path: Path = DEFAULT_MODEL_PATH) -> None:
    """Option 7: View model evaluation."""
    print_header("7. View Model Evaluation")

    if not model_path.is_file():
        print(f"[WARNING] No saved model artifact found at: '{model_path}'.")
        print("          Train a model using Menu Option 5 ('Train ML model') first.")
        return

    try:
        predictor = ConsumptionPredictor.load(model_path)
    except Exception as exc:
        print(f"\n[ERROR] Failed to load model artifact: {exc}")
        return

    meta = predictor.training_metadata
    print_section("Model Architecture & Metadata")
    print(f"  * Model Artifact Path        : {model_path.resolve()}")
    print(f"  * Model Type                 : {meta.get('model_type', type(predictor.model).__name__)}")
    print(f"  * Algorithm Description      : {meta.get('model_description', 'Linear regression baseline')}")
    print(f"  * Reference Base Date (Day 0): {meta.get('reference_date', 'N/A')}")
    print(f"  * Feature Name               : {meta.get('feature_name', 'day_index')}")
    print(f"  * Target Variable            : {predictor.target_name}")
    print(f"  * Training Timestamp         : {predictor.trained_at or 'Unknown'}")

    if meta:
        slope = meta.get("slope_kwh_per_day", 0.0)
        intercept = meta.get("intercept_kwh", 0.0)
        sign = "+" if slope >= 0 else "-"
        print(f"  * Fitted Formula             : usage_kwh = {intercept:.4f} {sign} {abs(slope):.4f} * day_index")
        print(f"  * Daily Trend Drift Rate     : {slope:+.4f} kWh/day")
        print(f"  * Train/Test Observations    : {meta.get('train_records', 'N/A')} train / {meta.get('test_records', 'N/A')} test")
        print(f"  * Train Date Span            : {meta.get('train_date_range', 'N/A')}")
        print(f"  * Test Date Span             : {meta.get('test_date_range', 'N/A')}")

    if predictor.evaluation_metrics:
        print_section("Out-of-Sample Forward Evaluation (Test Set)")
        mae = predictor.evaluation_metrics.get("mae")
        rmse = predictor.evaluation_metrics.get("rmse")
        r2 = predictor.evaluation_metrics.get("r2_score")
        if mae is not None:
            print_metric("Mean Absolute Error (MAE)", mae, unit="kWh", decimal_places=4)
        if rmse is not None:
            print_metric("Root Mean Squared Error (RMSE)", rmse, unit="kWh", decimal_places=4)
        if r2 is not None:
            print_metric("R^2 Score (Variance Explained)", r2, decimal_places=4)
        else:
            print("  * R^2 Score (Variance Explained) : Undefined (insufficient test variance)")
    else:
        print("\n[INFO] No evaluation metrics stored with this model.")

    print_section("Methodological Notice")
    print("  * This model is an interpretable trend-based baseline, not a seasonal forecasting system.")
    print("  * Metrics are computed directly on forward test data without data leakage or invented accuracy.")


def menu_exit() -> None:
    """Option 8: Exit."""
    print(f"\nThank you for using {APPLICATION_TITLE}. Goodbye!\n")
    sys.exit(0)


# =====================================================================
# Main Application Loop
# =====================================================================

def interactive_application_loop() -> None:
    """Run the main interactive menu loop continuously until option 8 is chosen."""
    while True:
        display_menu()

        choice = safe_input("Enter your choice [1-8]: ")

        # Handle Ctrl+C or EOF at main prompt
        if choice is None:
            menu_exit()

        # Handle blank input
        if choice == "":
            print("\n[WARNING] Blank input received. Please enter a valid menu number (1-8).")
            pause_for_user()
            continue

        # Dispatch valid menu options
        if choice == "1":
            menu_view_records()
        elif choice == "2":
            menu_add_record()
        elif choice == "3":
            menu_update_record()
        elif choice == "4":
            menu_view_statistics()
        elif choice == "5":
            menu_train_model()
        elif choice == "6":
            menu_predict_future()
        elif choice == "7":
            menu_view_model_evaluation()
        elif choice == "8":
            menu_exit()
        else:
            print(f"\n[ERROR] Invalid choice '{choice}'. Please select a valid number between 1 and 8.")

        pause_for_user()


# =====================================================================
# Command Line Argument Parser (Automation & Non-Interactive Mode)
# =====================================================================

def parse_cli_args() -> argparse.Namespace:
    """Parse optional command-line flags for script automation."""
    parser = argparse.ArgumentParser(
        description=f"{APPLICATION_TITLE} - Command Line Interface"
    )
    parser.add_argument("--train", action="store_true", help="Train and save ML model.")
    parser.add_argument("--summary", action="store_true", help="Display dataset statistics.")
    parser.add_argument(
        "--view-records",
        type=int,
        nargs="?",
        const=20,
        help="View latest records (default 20).",
    )
    parser.add_argument(
        "--add-record",
        nargs=2,
        metavar=("DATE", "USAGE_KWH"),
        help="Add record: --add-record 2025-01-01 24.5",
    )
    parser.add_argument(
        "--update-record",
        nargs=2,
        metavar=("DATE", "USAGE_KWH"),
        help="Update record: --update-record 2025-01-01 26.0",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Regenerate demonstration synthetic dataset.",
    )
    parser.add_argument(
        "--model-info", action="store_true", help="View saved model metrics."
    )
    parser.add_argument(
        "--predict", action="store_true", help="Predict future usage via flags."
    )
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD).")
    parser.add_argument("--lag1", type=float, help="Yesterday's usage in kWh.")
    parser.add_argument("--lag7", type=float, help="Last week same-day usage in kWh.")
    parser.add_argument("--rolling", type=float, help="7-day rolling average in kWh.")
    return parser.parse_args()


def main() -> None:
    """Main program entrypoint."""
    args = parse_cli_args()

    # If any CLI arguments are passed, execute that action and exit
    if args.generate_data:
        print_header("Generate Demonstration Sample Data")
        df = generate_synthetic_sample_data(allow_overwrite=True)
        print(f"[SUCCESS] Saved {len(df)} records to: {DEFAULT_DATA_PATH}")
        return
    if args.summary:
        menu_view_statistics()
        return
    if args.view_records is not None:
        df = ensure_dataset_exists()
        if df is not None:
            records = view_all_records(df, limit=args.view_records, tail=True)
            for idx, row in records.iterrows():
                print(f"  {idx + 1:<4} | {str(row['date'])[:10]} | {float(row[TARGET_COLUMN]):.2f} kWh")
        return
    if args.add_record:
        date_str, usage_str = args.add_record
        df = ensure_dataset_exists()
        if df is not None:
            try:
                val = float(usage_str)
                updated = add_record(df, date_str, val)
                save_data(updated, DEFAULT_DATA_PATH, allow_overwrite=True)
                print(f"[SUCCESS] Added record for {date_str}: {val:.2f} kWh.")
            except Exception as exc:
                print(f"[ERROR] Failed to add record: {exc}")
        return
    if args.update_record:
        date_str, usage_str = args.update_record
        df = ensure_dataset_exists()
        if df is not None:
            try:
                val = float(usage_str)
                updated = update_record(df, date_str, val)
                save_data(updated, DEFAULT_DATA_PATH, allow_overwrite=True)
                print(f"[SUCCESS] Updated record for {date_str}: {val:.2f} kWh.")
            except Exception as exc:
                print(f"[ERROR] Failed to update record: {exc}")
        return
    if args.train:
        menu_train_model()
        return
    if args.model_info:
        menu_view_model_evaluation()
        return
    if args.predict:
        target_date = args.date or datetime.now().strftime("%Y-%m-%d")
        lag1 = args.lag1 if args.lag1 is not None else 25.0
        lag7 = args.lag7 if args.lag7 is not None else 25.0
        rolling = args.rolling if args.rolling is not None else 25.0

        if not DEFAULT_MODEL_PATH.is_file():
            menu_train_model()
        predictor = ConsumptionPredictor.load(DEFAULT_MODEL_PATH)
        result = predictor.predict_for_date(
            date_input=target_date, lag_1=lag1, lag_7=lag7, rolling_7=rolling
        )
        print(f"Predicted Electricity Consumption for {target_date}: {result:.2f} kWh")
        return

    # No arguments provided: launch clean interactive application loop
    try:
        interactive_application_loop()
    except (KeyboardInterrupt, EOFError):
        print(f"\n\nExiting {APPLICATION_TITLE}. Goodbye!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
