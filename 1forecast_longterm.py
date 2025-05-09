import numpy as np
import pandas as pd
import joblib
import os
import warnings
import traceback
import math
import matplotlib.pyplot as plt
import time


PRETRAINED_MODEL_DIR = "pretrained_models_v22"

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
CYCLES_PER_PREDICTION_STEP = 500


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



DEFAULT_FORECAST_YEARS = 15
PLOT_RESULTS = True
MINIMUM_SOH_THRESHOLD = 60.0


def load_model(model_path, model_type):

    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path): print(f"ERROR: Model file not found."); return None
    try:
        if model_type == 'NN':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'; warnings.filterwarnings('ignore', category=FutureWarning)
            try:
                import tensorflow as tf; model = tf.keras.models.load_model(model_path, compile=False); return model
            except ImportError: print("ERROR: TensorFlow not found."); return None
            except Exception as e: print(f"ERROR loading Keras model: {e}"); return None
        else: return joblib.load(model_path)
    except Exception as e: print(f"ERROR loading model file: {e}"); return None

def load_scaler(scaler_path):

    print(f"Loading scaler: {scaler_path}")
    if not os.path.exists(scaler_path): print(f"ERROR: Scaler file not found."); return None
    try: return joblib.load(scaler_path)
    except Exception as e: print(f"ERROR loading scaler file: {e}"); return None


def get_user_scenario_for_forecast():
    """Gets user input for the forecasting scenario."""
    print("\n--- Define Scenario for Long-Term Forecast ---")
    user_choices = {}


    available_types = list(MODEL_CONFIG.keys())
    while True:
        print("\nSelect the battery type for the forecast:")
        for i, batt_type in enumerate(available_types): print(f"  {i+1}. {batt_type}")
        choice = input("Enter number: ").strip()
        try:
            index = int(choice) - 1
            if 0 <= index < len(available_types):
                user_choices['battery_type'] = available_types[index]
                break
            else: print("Invalid number.")
        except ValueError: print("Invalid input. Please enter a number.")


    print("\nSelect the climate for the forecast:")
    for key, value in CLIMATE_MAP.items():
        print(f"  {key}. {value['name']}")
        if key == "1": print("     (e.g., Oslo, Anchorage, Chicago/Minneapolis in winter)")
        if key == "2": print("     (e.g., London, New York, Los Angeles, Seattle)")
        if key == "3": print("     (e.g., Phoenix, Miami, Houston, Honolulu)")
        if key == "4": print("     (e.g., Extreme desert heat)")
    while True:
        choice = input("Select climate zone (enter number): ").strip()
        if choice in CLIMATE_MAP: user_choices['climate_choice'] = choice; break
        else: print("Invalid selection.")


    print("\nSelect the typical vehicle usage for the forecast:")
    for key, value in USAGE_PROFILE_MAP.items():
        print(f"  {key}. {value['name']}")
        if key == "1": print("     (Primarily short local trips, infrequent fast charging)")
        if key == "2": print("     (Mix of city/highway, regular charging, occasional DCFC)")
        if key == "3": print("     (Frequent long drives, relies often on DCFC)")
        if key == "4": print("     (Very high daily mileage, very frequent DCFC)")
    while True:
        choice = input("Select usage profile (enter number): ").strip()
        if choice in USAGE_PROFILE_MAP: user_choices['usage_profile_choice'] = choice; break
        else: print("Invalid selection.")


    print("\nSelect the typical traffic condition for the forecast:")
    for key, value in TRAFFIC_MAP.items():
        print(f"  {key}. {value['name']}")
        if key == "1": print("     (Mostly open roads, minimal congestion)")
        if key == "2": print("     (Some congestion, suburban/mixed driving)")
        if key == "3": print("     (Frequent heavy stop-and-go - dense city centers)")
    while True:
        choice = input("Select traffic condition (enter number): ").strip()
        if choice in TRAFFIC_MAP: user_choices['traffic_choice'] = choice; break
        else: print("Invalid selection.")


    print("\nEnter the starting State of Health (SoH) for the forecast:")
    print(" (Use 100 for a new battery, lower for used)")
    while True:
        try:
            soh_str = input("Enter starting SoH percentage (e.g., 100): ").strip()
            soh = float(soh_str)
            if 70.0 <= soh <= 100.0:
                user_choices['initial_soh_percent'] = soh
                break
            else: print("Please enter a percentage between 70 and 100.")
        except ValueError: print("Invalid input. Please enter a number.")


    while True:
        try:
            years_str = input(f"Enter forecast duration in years (e.g., {DEFAULT_FORECAST_YEARS}): ").strip()
            years = int(years_str)
            if 1 <= years <= 50:
                user_choices['forecast_years'] = years
                break
            else: print("Please enter a duration between 1 and 50 years.")
        except ValueError: print("Invalid input. Please enter a whole number of years.")

    print("--- Scenario Defined ---")
    return user_choices


def map_inputs_for_forecast(scenario_dict, target_feature_names):
    numerical_features = {}
    try:
        climate_key = scenario_dict['climate_choice']
        numerical_features['avg_temp_c'] = CLIMATE_MAP[climate_key]['temp_c']
        usage_key = scenario_dict['usage_profile_choice']
        profile_params = USAGE_PROFILE_MAP[usage_key]['params']
        for feature_name in ["dod", "charge_rate", "discharge_rate", "cycles_per_day"]:
            numerical_features[feature_name] = profile_params[feature_name]
        traffic_key = scenario_dict['traffic_choice']
        numerical_features['traffic_level'] = TRAFFIC_MAP[traffic_key]['level']
        if 'initial_soh_percent' not in scenario_dict: raise KeyError("'initial_soh_percent' key missing")
        static_features = {k: v for k, v in numerical_features.items()}
        static_feature_names = list(static_features.keys()) + ['initial_soh_percent']
        missing = [f for f in target_feature_names if f not in static_feature_names]
        if missing: print(f"ERROR: Mapping missing features: {', '.join(missing)}"); return None
        return numerical_features
    except KeyError as e: print(f"ERROR: Missing key in scenario/mapping: {e}"); return None
    except Exception as e: print(f"ERROR during input mapping: {e}"); return None


def predict_single_step_degradation(model, scaler, features_dict, feature_names, output_labels):
    if model is None or scaler is None or features_dict is None: return None
    try: features_df = pd.DataFrame([features_dict], columns=feature_names)
    except Exception as e: print(f"ERROR creating DataFrame: {e}"); return None
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore", category=UserWarning); features_scaled = scaler.transform(features_df)
    except Exception as e: print(f"ERROR scaling features: {e}"); return None
    try:
        predictions_scaled = model.predict(features_scaled, verbose=0);
        if predictions_scaled.ndim == 1: predictions_scaled = predictions_scaled.reshape(1, -1)
        num_outputs = predictions_scaled.shape[1]; results = {label: round(predictions_scaled[0, i], 2) for i, label in enumerate(output_labels) if i < num_outputs}
        if len(results) != len(output_labels): print(f"WARNING: Model outputs != Expected labels.")
        return results
    except Exception as e: print(f"ERROR during prediction: {e}"); return None


def plot_soh_forecast(forecast_data, scenario_name):
    if not forecast_data: print("No forecast data to plot."); return
    df = pd.DataFrame(forecast_data); plt.figure(figsize=(10, 6))
    plt.plot(df['Year'], df['SoH_Percent'], marker='o', linestyle='-', label='Predicted SoH')
    if df['SoH_Percent'].min() < 80: plt.axhline(y=80, color='r', linestyle='--', label='80% SoH Threshold (Typical EoL)')
    if df['SoH_Percent'].min() < 70: plt.axhline(y=70, color='orange', linestyle='--', label='70% SoH Threshold')
    plt.title(f'Forecasted Battery SoH Decline\nScenario: {scenario_name}')
    plt.xlabel('Year'); plt.ylabel('State of Health (%)')
    plt.ylim(min(MINIMUM_SOH_THRESHOLD - 5, df['SoH_Percent'].min() - 5), 105)
    plt.xlim(0, df['Year'].max() + 1); plt.grid(True, linestyle='--', alpha=0.7); plt.legend(); plt.tight_layout()
    try:
        filename = f"forecast_{scenario_name.replace(' ', '_').replace('/', '_').replace('(','').replace(')','')}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(filename); print(f"\nPlot saved as: {filename}"); plt.show()
    except Exception as e: print(f"Error saving/showing plot: {e}")
    finally: plt.close()



if __name__ == "__main__":
    print("--- Interactive Long-Term Battery Degradation Forecast ---")
    start_time = time.time()


    scenario = get_user_scenario_for_forecast()
    if scenario is None: print("Scenario definition failed. Exiting."); exit(1)


    batt_type = scenario.get("battery_type")
    forecast_years = scenario.get("forecast_years", DEFAULT_FORECAST_YEARS)


    scenario_name = f"{CLIMATE_MAP[scenario['climate_choice']]['name']} " \
                    f"{USAGE_PROFILE_MAP[scenario['usage_profile_choice']]['name']} " \
                    f"{batt_type} ({TRAFFIC_MAP[scenario['traffic_choice']]['name']} Traffic) " \
                    f"Start {scenario['initial_soh_percent']}% SoH"

    print(f"\nForecasting for Scenario: '{scenario_name}'")
    print(f"Using models from directory: '{PRETRAINED_MODEL_DIR}'")
    if not batt_type or batt_type not in MODEL_CONFIG:
        print(f"ERROR: Invalid 'battery_type' selected. Exiting."); exit(1)

    config = MODEL_CONFIG[batt_type]
    feature_names_for_model = config["feature_names"]


    print("\nLoading model and scaler...")
    model = load_model(os.path.join(PRETRAINED_MODEL_DIR, config["model_file"]), config["model_type"])
    scaler = load_scaler(os.path.join(PRETRAINED_MODEL_DIR, config["scaler_file"]))

    if not model or not scaler: print("ERROR: Failed to load model or scaler. Exiting."); exit(1)

    if hasattr(scaler, 'n_features_in_'):
        expected_features = len(config["feature_names"])
        if scaler.n_features_in_ != expected_features:
            print(f"CRITICAL ERROR: Scaler expects {scaler.n_features_in_} features, config expects {expected_features}. Exiting.")
            exit(1)
    print("Model and scaler loaded successfully.")


    base_numerical_features = map_inputs_for_forecast(scenario, feature_names_for_model)
    if base_numerical_features is None: print("ERROR: Failed to map scenario inputs. Exiting."); exit(1)


    try:
        cycles_per_day = base_numerical_features['cycles_per_day']
        cycles_per_year = cycles_per_day * 365.25
        total_forecast_cycles = cycles_per_year * forecast_years
        num_prediction_steps = math.ceil(total_forecast_cycles / CYCLES_PER_PREDICTION_STEP)
        print(f"\nForecasting Parameters:")
        print(f" - Duration: {forecast_years} years")
        print(f" - Cycles/Day: {cycles_per_day:.2f} (from profile: {USAGE_PROFILE_MAP[scenario['usage_profile_choice']]['name']})")
        print(f" - Cycles/Year: {cycles_per_year:.1f}")
        print(f" - Total Cycles to Simulate: {total_forecast_cycles:.0f}")
        print(f" - Cycles per Prediction Step: {CYCLES_PER_PREDICTION_STEP}")
        print(f" - Number of Prediction Steps: {num_prediction_steps}")
    except KeyError as e: print(f"ERROR: Missing parameter for cycle calculation: {e}. Exiting."); exit(1)
    except Exception as e: print(f"ERROR calculating forecast steps: {e}. Exiting."); exit(1)


    print("\n--- Starting Forecast Simulation ---")
    current_soh = scenario['initial_soh_percent']
    forecast_results = []

    forecast_results.append({
        "Step": 0, "Year": 0.0, "Cumulative_Cycles": 0,
        "SoH_Percent": current_soh, "ResistanceInc_Percent_Step": 0.0
    })

    total_cycles_simulated = 0
    for step in range(1, num_prediction_steps + 1):
        if current_soh < MINIMUM_SOH_THRESHOLD:
            print(f"INFO: SoH ({current_soh:.2f}%) reached minimum threshold ({MINIMUM_SOH_THRESHOLD}%) at step {step}. Stopping forecast.")
            break

        step_features = base_numerical_features.copy()
        step_features['initial_soh_percent'] = current_soh

        prediction = predict_single_step_degradation(
            model, scaler, step_features,
            feature_names_for_model, OUTPUT_LABELS
        )

        if prediction is None: print(f"ERROR: Prediction failed at step {step}. Stopping forecast."); break

        fade_this_step = prediction.get('capacity_fade_percent', 0)
        res_inc_this_step = prediction.get('resistance_increase_percent', 0)

        current_soh -= fade_this_step
        current_soh = max(0, current_soh)

        total_cycles_simulated += CYCLES_PER_PREDICTION_STEP
        current_year = total_cycles_simulated / cycles_per_year

        print(f"Step {step:02d}: Year ~{current_year:.2f} ({total_cycles_simulated} cyc) -> Fade: {fade_this_step:.2f}% | New SoH: {current_soh:.2f}% | Res Inc: {res_inc_this_step:.2f}%")

        forecast_results.append({
            "Step": step, "Year": current_year, "Cumulative_Cycles": total_cycles_simulated,
            "SoH_Percent": current_soh, "ResistanceInc_Percent_Step": res_inc_this_step
        })

    print("--- Forecast Simulation Finished ---")


    if forecast_results:
        final_result = forecast_results[-1]
        print("\n--- Forecast Summary ---")
        print(f"Scenario: {scenario_name}")
        print(f"Forecast Duration Run: {final_result['Year']:.2f} years ({final_result['Step']} steps)")
        print(f"Total Cycles Simulated: {final_result['Cumulative_Cycles']}")
        print(f"Starting SoH: {scenario['initial_soh_percent']:.2f}%")
        print(f"Ending SoH: {final_result['SoH_Percent']:.2f}%")
        print(f"Total Capacity Fade: {scenario['initial_soh_percent'] - final_result['SoH_Percent']:.2f}%")
    else:
        print("\n--- No forecast results generated. ---")



    if PLOT_RESULTS and forecast_results:
        print("\nGenerating plot...")
        plot_soh_forecast(forecast_results, scenario_name)
    elif PLOT_RESULTS:
        print("\nSkipping plot: No forecast data.")


    print(f"\nTotal script execution time: {time.time() - start_time:.2f} seconds")
    print("--- End of Script ---")