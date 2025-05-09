
import numpy as np
import pandas as pd
import joblib
import os
import warnings
import traceback


PRETRAINED_MODEL_DIR = "pretrained_models_v2"

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
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path): print(f"ERROR: Model file not found at {model_path}"); return None
    try:
        if model_type == 'NN':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'; warnings.filterwarnings('ignore', category=FutureWarning)
            try:
                import tensorflow as tf; model = tf.keras.models.load_model(model_path, compile=False); print("Keras model loaded successfully."); return model
            except ImportError: print("ERROR: TensorFlow not found."); return None
            except Exception as e: print(f"ERROR loading Keras model: {e}"); traceback.print_exc(); return None
        else: model = joblib.load(model_path); print("Joblib model loaded successfully."); return model
    except Exception as e: print(f"ERROR loading model file {model_path}: {e}"); traceback.print_exc(); return None

def load_scaler(scaler_path):
    print(f"Attempting to load scaler from: {scaler_path}")
    if not os.path.exists(scaler_path): print(f"ERROR: Scaler file not found at {scaler_path}"); return None
    try:
        scaler = joblib.load(scaler_path); print("Scaler loaded successfully.")
        if not hasattr(scaler, 'transform') or not hasattr(scaler, 'mean_'): print(f"WARNING: Loaded object might not be valid scaler.")

        return scaler
    except Exception as e: print(f"ERROR loading scaler file {scaler_path}: {e}"); traceback.print_exc(); return None



def get_user_input_simplified_v2():
    """Gets user input using simplified choices, including enhanced examples."""
    print("\n--- Tell Us About Your Scenario ---")
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
            else: print("Invalid number.")
        except ValueError: print("Invalid input. Please enter a number.")


    print("\nSelect the climate that best describes your location:")
    for key, value in CLIMATE_MAP.items():
        print(f"  {key}. {value['name']}")

        if key == "1": print("     (e.g., Oslo, Anchorage, Chicago/Minneapolis in winter, higher elevations like Denver)")
        if key == "2": print("     (e.g., London, New York, Los Angeles, Seattle, Dallas, most of Temperate US/Europe)")
        if key == "3": print("     (e.g., Phoenix, Miami, Houston, Honolulu - hotter parts of US, Southeast Asia, Middle East)")
        if key == "4": print("     (e.g., Extreme desert heat, Death Valley conditions)")
    while True:
        choice = input("Select climate zone (enter number): ").strip()
        if choice in CLIMATE_MAP:
            user_choices['climate_choice'] = choice
            break
        else: print("Invalid selection.")


    print("\nSelect the description that best fits your typical vehicle usage:")
    for key, value in USAGE_PROFILE_MAP.items():
        print(f"  {key}. {value['name']}")
        if key == "1": print("     (Primarily short local trips, infrequent fast charging)")
        if key == "2": print("     (Mix of city/highway, regular charging, occasional road trips/DCFC - typical personal use)")
        if key == "3": print("     (Frequent long drives, relies often on DC fast charging)")
        if key == "4": print("     (Very high daily mileage, extremely frequent DC fast charging, professional use)")
    while True:
        choice = input("Select usage profile (enter number): ").strip()
        if choice in USAGE_PROFILE_MAP:
            user_choices['usage_profile_choice'] = choice
            break
        else: print("Invalid selection.")


    print("\nSelect the typical traffic condition you drive in:")
    for key, value in TRAFFIC_MAP.items():
        print(f"  {key}. {value['name']}")
        if key == "1": print("     (Mostly open roads, minimal congestion)")
        if key == "2": print("     (Some congestion, suburban/mixed driving - common in many cities like Phoenix, Austin, Seattle suburbs)")
        if key == "3": print("     (Frequent heavy stop-and-go - typical for dense city centers like London, NYC, LA)")
    while True:
        choice = input("Select traffic condition (enter number): ").strip()
        if choice in TRAFFIC_MAP:
            user_choices['traffic_choice'] = choice
            break
        else: print("Invalid selection.")


    print("\nEstimate the battery's current State of Health (SoH):")
    print(" (100% = New, 90% = Some use/age, 80% = More aged)")
    while True:
        try:
            soh_str = input("Enter current SoH percentage (e.g., 95): ").strip()
            soh = float(soh_str)
            if 70.0 <= soh <= 100.0:
                user_choices['initial_soh_percent'] = soh
                break
            else: print("Please enter a percentage between 70 and 100.")
        except ValueError: print("Invalid input. Please enter a number.")

    print("--- Input Received ---")
    return user_choices



def map_simplified_inputs_v2(scenario_choices, target_feature_names):

    print("Mapping simplified choices (V2) to model features...")
    numerical_features = {}
    climate_key = scenario_choices.get('climate_choice')
    if climate_key and climate_key in CLIMATE_MAP: numerical_features['avg_temp_c'] = CLIMATE_MAP[climate_key]['temp_c']
    else: print("ERROR: Invalid/missing climate_choice."); return None
    usage_key = scenario_choices.get('usage_profile_choice')
    if usage_key and usage_key in USAGE_PROFILE_MAP:
        profile_params = USAGE_PROFILE_MAP[usage_key]['params']
        for feature_name in ["dod", "charge_rate", "discharge_rate", "cycles_per_day"]:
             if feature_name in profile_params: numerical_features[feature_name] = profile_params[feature_name]
             else: print(f"WARNING: Usage map missing '{feature_name}'."); numerical_features[feature_name] = 0.0
    else: print("ERROR: Invalid/missing usage_profile_choice."); return None
    traffic_key = scenario_choices.get('traffic_choice')
    if traffic_key and traffic_key in TRAFFIC_MAP: numerical_features['traffic_level'] = TRAFFIC_MAP[traffic_key]['level']
    else: print("ERROR: Invalid/missing traffic_choice."); return None
    soh_val = scenario_choices.get('initial_soh_percent')
    if soh_val is not None: numerical_features['initial_soh_percent'] = float(soh_val)
    else: print("ERROR: Missing initial_soh_percent."); return None
    missing = [f for f in target_feature_names if f not in numerical_features]
    if missing: print(f"ERROR: Mapping missing features: {', '.join(missing)}\nMapped: {numerical_features}"); return None
    print("V2 Input mapping complete.")
    return numerical_features


def predict_degradation(model, scaler, features_dict, feature_names, output_labels):
    if model is None or scaler is None or features_dict is None: print("ERROR: Model/scaler/features missing for prediction."); return None
    print("\n--- Predicting Degradation ---")
    try: features_df = pd.DataFrame([features_dict], columns=feature_names); print("Input DataFrame for prediction (mapped values):\n", features_df)
    except Exception as e: print(f"ERROR creating DataFrame: {e}\nRequired: {feature_names}\nProvided keys: {list(features_dict.keys())}"); return None
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore", category=UserWarning); features_scaled = scaler.transform(features_df)
        print("Scaled features (first row):", features_scaled[0])
    except Exception as e:
        print(f"ERROR scaling features: {e}")
        if hasattr(scaler, 'n_features_in_'): print("Scaler expected number of features:", scaler.n_features_in_)
        print("Actual features provided to scaler:", features_df.columns.tolist()); traceback.print_exc(); return None
    try:
        predictions_scaled = model.predict(features_scaled);
        if predictions_scaled.ndim == 1: predictions_scaled = predictions_scaled.reshape(1, -1)
        print("Raw model predictions:", predictions_scaled[0])
        num_outputs = predictions_scaled.shape[1]; results = {label: round(predictions_scaled[0, i], 2) for i, label in enumerate(output_labels) if i < num_outputs}
        if len(results) != len(output_labels): print(f"WARNING: Model outputs ({num_outputs}) != Expected labels ({len(output_labels)}).")
        return results
    except Exception as e: print(f"ERROR during model prediction: {e}"); traceback.print_exc(); return None



def print_charging_interpretation(predicted_fade, predicted_res_inc, initial_soh, final_soh_est):
    """Prints the interpretation of degradation results on charging."""
    print("\n--- What This Means for Charging ---")
    print("\n*   Capacity Fade Impact:")
    if predicted_fade > 0 and isinstance(final_soh_est, (int, float)):
        print(f"    - Reduced Range: Vehicle's range will decrease.")
        original_range_example = 300
        final_range_est = original_range_example * (final_soh_est / 100.0)
        print(f"      (Example: If original range was {original_range_example}, SoH of {final_soh_est:.1f}% gives approx. {final_range_est:.0f} range).")
        print(f"    - More Frequent Charging: Likely need to charge more often.")
    elif predicted_fade > 0:
        print(f"    - Reduced Range: Vehicle's range will decrease.")
        print(f"    - More Frequent Charging: Likely need to charge more often.")
    else:
        print("    - Minimal capacity fade predicted for this period.")
    print("\n*   Resistance Increase Impact:")
    if predicted_res_inc > 5:
        print(f"    - Slower Peak Charging: Increased resistance generates heat. BMS may reduce charging speeds,")
        print(f"      especially during DC Fast Charging (DCFC), to protect the battery (esp. at high SoC/temp).")
        print(f"    - Charging time might increase slightly.")
    elif predicted_res_inc > 0: print("    - Slight increase in resistance. May have minimal noticeable impact on charging speeds.")
    else: print("    - Minimal resistance increase predicted for this period.")
    print("\n*   Overall:")
    print(f"    - Expect to adjust charging habits (more frequent/longer stops) compared to a new battery.")
    print("\n*   Disclaimer:")
    print("    - Range example uses linear scaling & assumed original range.")
    print("    - Charging speed impact is complex (vehicle/charger/temp dependent).")



if __name__ == "__main__":
    print("--- Interactive Battery Degradation Predictor (V2 - Context-Aware Prompts) ---")
    print(f"Loading V2 models/scalers from: '{PRETRAINED_MODEL_DIR}'")
    print(f"Predicting additional degradation over approx. {TRAINING_TARGET_CYCLES} future cycles.")


    user_scenario_choices = get_user_input_simplified_v2()

    if user_scenario_choices is None: print("Exiting due to input error or cancellation."); exit(1)
    selected_battery_type = user_scenario_choices.get('battery_type')
    if not selected_battery_type or selected_battery_type not in MODEL_CONFIG: print(f"ERROR: Invalid/missing battery type: {selected_battery_type}"); exit(1)


    config = MODEL_CONFIG[selected_battery_type]
    model_path = os.path.join(PRETRAINED_MODEL_DIR, config["model_file"])
    scaler_path = os.path.join(PRETRAINED_MODEL_DIR, config["scaler_file"])

    model = load_model(model_path, config["model_type"])
    scaler = load_scaler(scaler_path)


    if scaler and hasattr(scaler, 'n_features_in_'):
        expected_features = len(config["feature_names"])
        if scaler.n_features_in_ != expected_features:
            print(f"\nCRITICAL WARNING: Loaded scaler expects {scaler.n_features_in_} features, "
                   f"but script config for {selected_battery_type} expects {expected_features}!")
            print("Ensure you copied the correct V2 scaler file. Exiting.")
            exit(1)
    elif scaler:
        print("\nWARNING: Loaded scaler object doesn't report expected features. Proceeding cautiously.")



    if model is None or scaler is None: print(f"\nERROR: Failed to load V2 model/scaler.\nEnsure V2 files ('{config['model_file']}', '{config['scaler_file']}') exist in '{PRETRAINED_MODEL_DIR}'."); exit(1)


    feature_names_for_model = config["feature_names"]
    numerical_features = map_simplified_inputs_v2(user_scenario_choices, feature_names_for_model)

    if numerical_features is None: print("Exiting due to input mapping error."); exit(1)


    predicted_results = predict_degradation(model, scaler, numerical_features, feature_names_for_model, OUTPUT_LABELS)


    print("\n--- Prediction Results & Interpretation ---")
    if predicted_results:
        print(f"Based on your selections:")
        print(f"  - Battery Type: {user_scenario_choices.get('battery_type', 'N/A')}")
        climate_name = CLIMATE_MAP.get(user_scenario_choices.get('climate_choice'), {}).get('name', 'N/A')
        print(f"  - Climate Zone: {climate_name}")
        usage_name = USAGE_PROFILE_MAP.get(user_scenario_choices.get('usage_profile_choice'), {}).get('name', 'N/A')
        print(f"  - Usage Profile: {usage_name}")
        traffic_name = TRAFFIC_MAP.get(user_scenario_choices.get('traffic_choice'), {}).get('name', 'N/A')
        print(f"  - Traffic Condition: {traffic_name}")
        initial_soh = numerical_features.get('initial_soh_percent', 'N/A')
        print(f"  - Starting SoH: {initial_soh}%")

        print(f"\nPredicted *Additional* Degradation (next {TRAINING_TARGET_CYCLES} cycles):")
        predicted_fade = predicted_results.get('capacity_fade_percent', 0)
        predicted_res_inc = predicted_results.get('resistance_increase_percent', 0)
        print(f"  - Capacity Fade Percent: {predicted_fade:.2f} %")
        print(f"  - Resistance Increase Percent: {predicted_res_inc:.2f} %")

        final_soh_est = 'N/A'
        if isinstance(initial_soh, (int, float)) and isinstance(predicted_fade, (int, float)):
             final_soh_est = initial_soh - predicted_fade
             print(f"\n  -> Estimated Final SoH after {TRAINING_TARGET_CYCLES} cycles: {final_soh_est:.2f}%")


        print_charging_interpretation(predicted_fade, predicted_res_inc, initial_soh, final_soh_est)


        print("\nDisclaimer:")
        print("  * This prediction uses generalized profiles and heuristic adjustments in the simulation.")
        print("  * It forecasts *additional* degradation from the specified starting SoH.")
        print("  * Actual real-world degradation can vary significantly.")

    else:
        print("Prediction failed.")

    print("\n--- End of Prediction ---")
