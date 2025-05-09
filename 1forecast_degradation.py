import numpy as np
import pandas as pd
import joblib
import os
import warnings
import traceback
import math
import matplotlib.pyplot as plt


PRETRAINED_MODEL_DIR = "pretrained_models_v2"
OUTPUT_DIR = "interactive_forecast_outputs"

MODEL_CONFIG = {
    "LiIon": {
        "model_file": "LiIon_model_v2.keras",
        "scaler_file": "LiIon_scaler_v2.joblib",
        "model_type": "NN",
        "feature_names": [
            "avg_temp_c", "dod", "charge_rate", "discharge_rate",
            "cycles_per_day", "initial_soh_percent", "traffic_level"
        ]
    },
    "NaIon": {
        "model_file": "NaIon_model_v2.keras",
        "scaler_file": "NaIon_scaler_v2.joblib",
        "model_type": "NN",
        "feature_names": [
            "avg_temp_c", "dod", "charge_rate", "discharge_rate",
            "cycles_per_day", "initial_soh_percent", "traffic_level"
        ]
    }
}

OUTPUT_LABELS = ["capacity_fade_percent", "resistance_increase_percent"]
TRAINING_TARGET_CYCLES = 500
FORECAST_YEARS = 15
SOH_EOL_THRESHOLD = 70.0


ASSUMED_INITIAL_RANGE_MILES = 300
ASSUMED_AVG_DAILY_MILES = {
    "1": 25, "2": 40, "3": 55, "4": 100
}
ASSUMED_CHARGE_TRIGGER_SOC = 20
ASSUMED_CHARGE_TARGET_SOC = 85


CLIMATE_MAP = {
    "1": {"name": "Cold (<10°C avg)", "temp_c": 5.0},
    "2": {"name": "Temperate (10-25°C avg)", "temp_c": 18.0},
    "3": {"name": "Hot (25-35°C avg)", "temp_c": 30.0},
    "4": {"name": "Very Hot (>35°C avg)", "temp_c": 40.0},
}
USAGE_PROFILE_MAP = {
    "1": {"name": "Gentle Commuter", "params": {"dod": 0.5, "charge_rate": 0.3, "discharge_rate": 0.5, "cycles_per_day": 0.8}},
    "2": {"name": "Average User", "params": {"dod": 0.7, "charge_rate": 0.7, "discharge_rate": 0.8, "cycles_per_day": 1.0}},
    "3": {"name": "Heavy User / Road Tripper", "params": {"dod": 0.85, "charge_rate": 1.5, "discharge_rate": 1.2, "cycles_per_day": 1.2}},
    "4": {"name": "Taxi / Ride Share", "params": {"dod": 0.9, "charge_rate": 1.8, "discharge_rate": 1.0, "cycles_per_day": 2.0}},
}
TRAFFIC_MAP = {
    "1": {"name": "Low / Highway", "level": 0},
    "2": {"name": "Medium / Mixed", "level": 1},
    "3": {"name": "High / City Stop-Go", "level": 2},
}


def load_model(model_path, model_type):
    """Loads a pre-trained model."""
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path): print(f"ERROR: Model file not found at {model_path}"); return None
    try:
        if model_type == 'NN':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'; warnings.filterwarnings('ignore', category=FutureWarning)
            try: import tensorflow as tf; model = tf.keras.models.load_model(model_path, compile=False); print("  OK: Keras model loaded."); return model
            except ImportError: print("ERROR: TensorFlow not found."); return None
            except Exception as e: print(f"ERROR loading Keras model: {e}"); return None
        else: model = joblib.load(model_path); print("  OK: Joblib model loaded."); return model
    except Exception as e: print(f"ERROR loading model file {model_path}: {e}"); return None

def load_scaler(scaler_path):
    """Loads a pre-trained StandardScaler."""
    print(f"Attempting to load scaler from: {scaler_path}")
    if not os.path.exists(scaler_path): print(f"ERROR: Scaler file not found at {scaler_path}"); return None
    try:
        scaler = joblib.load(scaler_path); print("  OK: Scaler loaded.")
        if not hasattr(scaler, 'transform') or not hasattr(scaler, 'mean_'): print(f"WARNING: Loaded object might not be valid scaler.")
        return scaler
    except Exception as e: print(f"ERROR loading scaler file {scaler_path}: {e}"); return None


def get_user_input_for_forecast():
    """Gets user input using simplified choices for the forecast STARTING point."""
    print("\n--- Define Starting Scenario for Forecast ---")
    user_choices = {}


    available_types = list(MODEL_CONFIG.keys())
    while True:
        print("\nSelect the battery type:")
        for i, batt_type in enumerate(available_types):
            print(f"  {i+1}. {batt_type}")
        choice = input("Enter number: ").strip()
        try:
            index = int(choice) - 1
            if 0 <= index < len(available_types):
                user_choices['battery_type'] = available_types[index]
                break
            else:
                print("Invalid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")


    print("\nSelect the climate for your location:")
    for key, value in CLIMATE_MAP.items():
        print(f"  {key}. {value['name']}")
        if key == "1": print("     (e.g., Oslo, Anchorage, Chicago/winter)")
        if key == "2": print("     (e.g., London, NY, LA, Seattle, Dallas)")
        if key == "3": print("     (e.g., Phoenix, Miami, Houston, Honolulu)")
        if key == "4": print("     (e.g., Extreme desert heat)")
    while True:
        choice = input("Select climate zone: ").strip()
        if choice in CLIMATE_MAP:
            user_choices['climate_choice'] = choice
            break
        else:
            print("Invalid selection.")


    print("\nSelect your typical vehicle usage:")
    for key, value in USAGE_PROFILE_MAP.items():
        print(f"  {key}. {value['name']}")
        if key == "1": print("     (Short local trips, infrequent fast charging)")
        if key == "2": print("     (Mixed city/highway, regular charging, occasional DCFC)")
        if key == "3": print("     (Frequent long drives, relies often on DCFC)")
        if key == "4": print("     (Very high mileage, extremely frequent DCFC, professional use)")
    while True:
        choice = input("Select usage profile: ").strip()
        if choice in USAGE_PROFILE_MAP:
            user_choices['usage_profile_choice'] = choice
            break
        else:
            print("Invalid selection.")


    print("\nSelect your typical traffic condition:")
    for key, value in TRAFFIC_MAP.items():
        print(f"  {key}. {value['name']}")
        if key == "1": print("     (Mostly open roads, minimal congestion)")
        if key == "2": print("     (Some congestion, suburban/mixed - common)")
        if key == "3": print("     (Frequent heavy stop-and-go - dense city centers)")
    while True:
        choice = input("Select traffic condition: ").strip()
        if choice in TRAFFIC_MAP:
            user_choices['traffic_choice'] = choice
            break
        else:
            print("Invalid selection.")


    print("\nEnter the battery's STARTING State of Health (SoH) for the forecast:")
    print(" (Use 100% for a new battery, lower if modeling an older one)")
    while True:
        try:
            soh_str = input("Enter STARTING SoH percentage (e.g., 100): ").strip()
            soh = float(soh_str)
            if 70.0 <= soh <= 100.0:
                user_choices['initial_soh_percent'] = soh
                break
            else:
                print("Please enter a percentage between 70 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("--- Starting Scenario Input Received ---")
    return user_choices


def map_user_input_to_base_features(user_choices, target_feature_names):
    """Maps the user's choices to the base numerical features."""
    print("Mapping user choices to base numerical features...")
    numerical_features = {}
    climate_key = user_choices.get('climate_choice')
    if climate_key and climate_key in CLIMATE_MAP: numerical_features['avg_temp_c'] = CLIMATE_MAP[climate_key]['temp_c']
    else: print("ERROR: Invalid/missing climate_choice."); return None
    usage_key = user_choices.get('usage_profile_choice')
    if usage_key and usage_key in USAGE_PROFILE_MAP:
        profile_params = USAGE_PROFILE_MAP[usage_key]['params']
        for feature_name in ["dod", "charge_rate", "discharge_rate", "cycles_per_day"]:
             if feature_name in profile_params: numerical_features[feature_name] = profile_params[feature_name]
             else: print(f"WARN: Usage map missing '{feature_name}'."); numerical_features[feature_name] = 0.0
    else: print("ERROR: Invalid/missing usage_profile_choice."); return None
    traffic_key = user_choices.get('traffic_choice')
    if traffic_key and traffic_key in TRAFFIC_MAP: numerical_features['traffic_level'] = TRAFFIC_MAP[traffic_key]['level']
    else: print("ERROR: Invalid/missing traffic_choice."); return None
    soh_val = user_choices.get('initial_soh_percent')
    if soh_val is not None: numerical_features['initial_soh_percent'] = float(soh_val)
    else: print("ERROR: Missing initial_soh_percent."); return None
    base_features = set(numerical_features.keys())
    target_set = set(target_feature_names)
    if not target_set.issubset(base_features):
         missing = list(target_set - base_features)
         print(f"ERROR: Base mapping missing: {', '.join(missing)}"); return None
    print("Base features mapped successfully.")
    return numerical_features


def predict_single_step_fade(model, scaler, features_dict, feature_names):
    """Runs one prediction step and returns only the capacity fade."""
    if model is None or scaler is None or features_dict is None: return None
    try: features_df = pd.DataFrame([features_dict], columns=feature_names)
    except Exception: return None
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore", category=UserWarning); features_scaled = scaler.transform(features_df)
    except Exception: return None
    try:
        predictions_scaled = model.predict(features_scaled, verbose=0);
        if predictions_scaled.ndim == 1: predictions_scaled = predictions_scaled.reshape(1, -1)
        if "capacity_fade_percent" in OUTPUT_LABELS:
            fade_index = OUTPUT_LABELS.index("capacity_fade_percent")
            if predictions_scaled.shape[1] > fade_index: return predictions_scaled[0, fade_index]
            else: print("WARN: Pred shape mismatch."); return 0
        else: print("WARN: 'capacity_fade_percent' not in labels."); return 0
    except Exception as e: print(f"ERROR prediction step: {e}"); return None


def run_forecast_loop(base_features, starting_soh, usage_profile_key, model, scaler, feature_names, forecast_years, cycles_per_step, soh_eol):
    """Performs the iterative forecast loop."""
    print(f"\n--- Running Forecast ---")
    print(f"  Starting SoH: {starting_soh:.1f}%, Duration: {forecast_years} years")
    if not usage_profile_key or usage_profile_key not in USAGE_PROFILE_MAP: print("ERROR: Invalid usage profile key."); return None, None
    cycles_per_day = USAGE_PROFILE_MAP[usage_profile_key]['params'].get('cycles_per_day', 1.0)
    if cycles_per_day <= 0: print("ERROR: Cycles/day <= 0."); return None, None
    avg_daily_miles = ASSUMED_AVG_DAILY_MILES.get(usage_profile_key, 40)
    print(f"  Using {cycles_per_day:.1f} cycles/day, {avg_daily_miles} avg daily miles.")
    years_per_step = cycles_per_step / (cycles_per_day * 365.25)
    print(f"  Time per {cycles_per_step}-cycle step: {years_per_step:.2f} years.")

    current_soh = starting_soh; current_year = 0.0
    forecast_results = []; time_to_eol = None
    step = 0
    while current_year < forecast_years:
        step += 1; start_soh_step = current_soh

        current_range_miles = ASSUMED_INITIAL_RANGE_MILES * (start_soh_step / 100.0)
        usable_soc_swing = ASSUMED_CHARGE_TARGET_SOC - ASSUMED_CHARGE_TRIGGER_SOC
        drivable_range_per_charge = current_range_miles * (usable_soc_swing / 100.0)
        if avg_daily_miles > 0 and drivable_range_per_charge > 0: days_between_charges = drivable_range_per_charge / avg_daily_miles
        else: days_between_charges = float('inf')
        forecast_results.append((current_year, start_soh_step, current_range_miles, days_between_charges))

        if start_soh_step < soh_eol and time_to_eol is None:
            print(f"  INFO: SoH below {soh_eol}% at start of step {step} ({current_year:.2f} years)."); time_to_eol = current_year

        step_features = base_features.copy(); step_features['initial_soh_percent'] = start_soh_step
        predicted_fade = predict_single_step_fade(model, scaler, step_features, feature_names)
        if predicted_fade is None: print(f"ERROR: Prediction failed step {step}. Stopping."); break
        if predicted_fade < 0: predicted_fade = 0.0

        current_soh -= predicted_fade; current_year += years_per_step

        if current_year >= forecast_years:
             final_range_miles = ASSUMED_INITIAL_RANGE_MILES * (max(0, current_soh) / 100.0)
             drivable_range_per_charge = final_range_miles * (usable_soc_swing / 100.0)
             if avg_daily_miles > 0 and drivable_range_per_charge > 0: final_days_between_charges = drivable_range_per_charge / avg_daily_miles
             else: final_days_between_charges = float('inf')
             forecast_results.append((current_year, max(0, current_soh), final_range_miles, final_days_between_charges))

        if current_soh < soh_eol and time_to_eol is None:
            print(f"  INFO: SoH crossed {soh_eol}% threshold during step {step} (around {current_year:.2f} years).")
            if predicted_fade > 0:
                 fade_fraction = max(0, (start_soh_step - soh_eol)) / predicted_fade
                 time_to_eol = (current_year - years_per_step) + (fade_fraction * years_per_step)
                 print(f"    (Interpolated EoL time: {time_to_eol:.2f} years)")
            else: time_to_eol = current_year

    if time_to_eol is None and current_year >= forecast_years:
         print(f"  INFO: SoH stayed above {soh_eol}% for {forecast_years} years."); time_to_eol = float('inf')
    print("--- Forecast Loop Finished ---")
    return forecast_results, time_to_eol


def plot_single_forecast(results_list, time_to_eol, scenario_details_str, filename="interactive_forecast_plot.png"):
    """Plots SoH vs Years and Range vs Years for a single scenario."""
    if not results_list: print("No forecast results to plot."); return

    years = [r[0] for r in results_list]
    soh = [r[1] for r in results_list]
    range_miles = [r[2] for r in results_list]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    fig.suptitle(f"Forecast for: {scenario_details_str}\n(Max {FORECAST_YEARS} Years, Based on {TRAINING_TARGET_CYCLES} Cycle Steps)", fontsize=14)
    ax_soh, ax_range = axes[0], axes[1]


    ax_soh.plot(years, soh, marker='.', linestyle='-', label="Forecasted SoH")
    ax_soh.axhline(y=SOH_EOL_THRESHOLD, color='r', linestyle='--', label=f'EOL Threshold ({SOH_EOL_THRESHOLD}%)')
    if time_to_eol is not None and time_to_eol <= FORECAST_YEARS:
         ax_soh.plot(time_to_eol, SOH_EOL_THRESHOLD, marker='X', markersize=10, color='red', linestyle='')
         ax_soh.text(time_to_eol, SOH_EOL_THRESHOLD + 1, f'{time_to_eol:.1f} yrs EoL', color='red', ha='center', va='bottom')
    ax_soh.set_ylabel("State of Health (SoH %)")
    ax_soh.set_ylim(bottom=max(0, SOH_EOL_THRESHOLD - 15), top=105)
    ax_soh.grid(True, linestyle='--'); ax_soh.legend()


    ax_range.plot(years, range_miles, marker='.', linestyle='-', label="Estimated Range", color='green')
    if time_to_eol is not None and time_to_eol <= FORECAST_YEARS:
         eol_range = ASSUMED_INITIAL_RANGE_MILES * (SOH_EOL_THRESHOLD / 100.0)
         ax_range.plot(time_to_eol, eol_range, marker='X', markersize=10, color='red', linestyle='')

         ax_range.text(time_to_eol, eol_range + (ASSUMED_INITIAL_RANGE_MILES * 0.02), f'{time_to_eol:.1f} yrs EoL', color='red', ha='center', va='bottom')
    ax_range.set_xlabel("Years")
    ax_range.set_ylabel(f"Estimated Range ({ASSUMED_INITIAL_RANGE_MILES} miles new)")

    ax_range.set_ylim(bottom=0, top=ASSUMED_INITIAL_RANGE_MILES * 1.05)

    ax_range.grid(True, linestyle='--'); ax_range.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR); print(f"Created output directory: {OUTPUT_DIR}")
    plot_filepath = os.path.join(OUTPUT_DIR, filename)
    try: plt.savefig(plot_filepath); print(f"\n--- Forecast plot saved as: {os.path.abspath(plot_filepath)} ---")
    except Exception as e: print(f"ERROR saving plot '{plot_filepath}': {e}")
    plt.close(fig)


if __name__ == "__main__":
    print("--- Interactive Battery Degradation Forecaster (V2) ---")

    user_choices = get_user_input_for_forecast()
    if user_choices is None: exit(1)
    selected_battery_type = user_choices.get("battery_type")
    starting_soh = user_choices.get("initial_soh_percent")
    usage_profile_key = user_choices.get("usage_profile_choice")


    print("\n--- Loading Model & Scaler ---")
    if selected_battery_type not in MODEL_CONFIG: print(f"ERROR: Config missing for {selected_battery_type}"); exit(1)
    config = MODEL_CONFIG[selected_battery_type]
    model_path = os.path.join(PRETRAINED_MODEL_DIR, config["model_file"])
    scaler_path = os.path.join(PRETRAINED_MODEL_DIR, config["scaler_file"])
    model = load_model(model_path, config["model_type"])
    scaler = load_scaler(scaler_path)
    if not model or not scaler: print("ERROR: Load failed. Exiting."); exit(1)
    if hasattr(scaler, 'n_features_in_'):
        expected_features = len(config["feature_names"])
        if scaler.n_features_in_ != expected_features:
            print(f"ERROR: Scaler features mismatch! Expected {expected_features}, got {scaler.n_features_in_}. Exiting."); exit(1)
    else: print("WARN: Cannot verify scaler feature count.")
    print("--- Models & Scalers Loaded ---")


    feature_names = config["feature_names"]
    base_features = map_user_input_to_base_features(user_choices, feature_names)
    if base_features is None: print("ERROR: Input mapping failed. Exiting."); exit(1)


    forecast_results, time_to_eol = run_forecast_loop(
        base_features=base_features, starting_soh=starting_soh, usage_profile_key=usage_profile_key,
        model=model, scaler=scaler, feature_names=feature_names,
        forecast_years=FORECAST_YEARS, cycles_per_step=TRAINING_TARGET_CYCLES, soh_eol=SOH_EOL_THRESHOLD
    )


    if forecast_results:
        print(f"\n--- Forecast Summary for Your Scenario ---")
        print("  " + "-"*65)
        print(f"  {'Year':<8} | {'SoH (%)':<8} | {'Range (mi)':<11} | {'Charge Every (days)':<20}")
        print("  " + "-"*65)
        for year, soh, range_miles, days_between_charges in forecast_results:
             charge_freq_str = f"{days_between_charges:.1f}" if days_between_charges != float('inf') else "N/A"
             print(f"  {year:<8.2f} | {soh:>8.2f} | {range_miles:>11.0f} | {charge_freq_str:<20}")
        print("  " + "-"*65)
        if time_to_eol is not None and time_to_eol <= FORECAST_YEARS: print(f"  Estimated Time to Reach {SOH_EOL_THRESHOLD}% SoH: {time_to_eol:.2f} years")
        else: print(f"  SoH remained above {SOH_EOL_THRESHOLD}% for the full {FORECAST_YEARS} year forecast.")

        clim_str = CLIMATE_MAP.get(user_choices.get("climate_choice"),{}).get("name","?")
        use_str = USAGE_PROFILE_MAP.get(usage_profile_key,{}).get("name","?")
        traf_str = TRAFFIC_MAP.get(user_choices.get("traffic_choice"),{}).get("name","?")
        scenario_details = f"{selected_battery_type} | Start {starting_soh:.0f}% SoH | {clim_str} | {use_str} | {traf_str} Traffic"

        plot_filename = f"forecast_{selected_battery_type}_start{starting_soh:.0f}soh_{clim_str}_{use_str}_{traf_str}.png"
        safe_chars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        plot_filename = "".join(c for c in plot_filename if c in safe_chars).replace(' ','_')

        plot_single_forecast(forecast_results, time_to_eol, scenario_details, filename=plot_filename)
    else: print("\nERROR: Forecast could not be completed.")

    print(f"\n{'='*60}")
    print("--- Interactive Forecasting Finished ---")