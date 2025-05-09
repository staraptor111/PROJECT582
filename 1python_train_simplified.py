
import numpy as np
import pandas as pd
import joblib
import os
import warnings
import traceback
import time

PRETRAINED_MODEL_DIR = "pretrained_models_v22"
CITY_DATA_CSV = "city_temperatures_clean.csv"
OUTPUT_RESULTS_CSV = "city_degradation_predictions_v22.csv"

DEFAULT_USAGE_PROFILE_CHOICE = "2" 
DEFAULT_INITIAL_SOH = 95.0        
BATTERY_TYPES_TO_RUN = ["LiIon", "NaIon"] 

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
    "1": {"name": "Cold (<10C)", "temp_c": 5.0},
    "2": {"name": "Temperate (10-25C)", "temp_c": 18.0},
    "3": {"name": "Hot (25-35C)", "temp_c": 30.0},
    "4": {"name": "Very Hot (>35C)", "temp_c": 40.0}
}
USAGE_PROFILE_MAP = {
    "1": {"name": "Gentle", "params": {"dod": 0.5, "charge_rate": 0.3, "discharge_rate": 0.5, "cycles_per_day": 0.8}},
    "2": {"name": "Average", "params": {"dod": 0.7, "charge_rate": 0.7, "discharge_rate": 0.8, "cycles_per_day": 1.0}},
    "3": {"name": "Heavy", "params": {"dod": 0.85, "charge_rate": 1.5, "discharge_rate": 1.2, "cycles_per_day": 1.2}},
    "4": {"name": "Taxi", "params": {"dod": 0.9, "charge_rate": 1.8, "discharge_rate": 1.0, "cycles_per_day": 2.0}}
}
TRAFFIC_MAP = {
    "1": {"name": "Low (Pop < 200k)", "level": 0},
    "2": {"name": "Medium (Pop 200k-1M)", "level": 1},
    "3": {"name": "High (Pop > 1M)", "level": 2}
}

def map_temp_to_climate_choice(temp_c):
    if pd.isna(temp_c):
        return None 
    if temp_c < 10.0: return "1"
    elif 10.0 <= temp_c < 25.0: return "2"
    elif 25.0 <= temp_c < 35.0: return "3"
    else: return "4" 

def map_pop_to_traffic_choice(population):
    if pd.isna(population):
        return "2" 
    if population > 1000000: return "3" 
    elif population > 200000: return "2" 
    else: return "1" 

def load_model(model_path, model_type):
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return None
    try:
        if model_type == 'NN':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
            warnings.filterwarnings('ignore', category=FutureWarning)
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path, compile=False)
                return model
            except ImportError:
                print("ERROR: TensorFlow not found. Cannot load NN model.")
                return None
            except Exception as e:
                print(f"ERROR loading Keras model '{model_path}': {e}")
                return None
        else: 
            model = joblib.load(model_path)
            return model
    except Exception as e:
        print(f"ERROR loading model file '{model_path}': {e}")
        return None

def load_scaler(scaler_path, expected_features):
    if not os.path.exists(scaler_path):
        print(f"ERROR: Scaler file not found: {scaler_path}")
        return None
    try:
        scaler = joblib.load(scaler_path)
        if not hasattr(scaler, 'transform') or not hasattr(scaler, 'mean_'):
             print(f"WARNING: Loaded object from {scaler_path} might not be a valid (fitted) scaler.")
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != expected_features:
             print(f"CRITICAL WARNING: Loaded scaler '{scaler_path}' expects {scaler.n_features_in_} features, "
                   f"but script config expects {expected_features}! Check model/scaler versions.")
        elif not hasattr(scaler, 'n_features_in_'):
             print(f"WARNING: Cannot verify feature count for scaler '{scaler_path}'.")
        return scaler
    except Exception as e:
        print(f"ERROR loading scaler file '{scaler_path}': {e}")
        return None

def map_scenario_to_features(scenario_dict, target_feature_names):
    numerical_features = {}

    climate_key = scenario_dict.get('climate_choice')
    if climate_key and climate_key in CLIMATE_MAP:
        numerical_features['avg_temp_c'] = CLIMATE_MAP[climate_key]['temp_c']
    else:
        print(f"INTERNAL ERROR: Invalid climate_choice '{climate_key}' passed to mapping function.")
        return None

    usage_key = scenario_dict.get('usage_profile_choice')
    if usage_key and usage_key in USAGE_PROFILE_MAP:
        profile_params = USAGE_PROFILE_MAP[usage_key]['params']
        for feature_name in ["dod", "charge_rate", "discharge_rate", "cycles_per_day"]:
            if feature_name in profile_params:
                numerical_features[feature_name] = profile_params[feature_name]
            else:
                print(f"WARNING: Assumed usage profile map is missing expected key '{feature_name}'. Setting to 0.0.")
                numerical_features[feature_name] = 0.0
    else:
        print(f"INTERNAL ERROR: Invalid usage_profile_choice '{usage_key}' passed to mapping function.")
        return None

    traffic_key = scenario_dict.get('traffic_choice')
    if traffic_key and traffic_key in TRAFFIC_MAP:
        numerical_features['traffic_level'] = TRAFFIC_MAP[traffic_key]['level']
    else:
        print(f"INTERNAL ERROR: Invalid traffic_choice '{traffic_key}' passed to mapping function.")
        return None

    soh_val = scenario_dict.get('initial_soh_percent')
    if soh_val is not None:
        try:
            numerical_features['initial_soh_percent'] = float(soh_val)
        except ValueError:
             print(f"ERROR: Invalid non-numeric initial_soh_percent '{soh_val}' in scenario.")
             return None
    else:
        print("ERROR: Missing initial_soh_percent in scenario dictionary.")
        return None

    missing = [f for f in target_feature_names if f not in numerical_features]
    if missing:
        print(f"ERROR: After mapping, the following required features are missing: {', '.join(missing)}")
        print(f"       Required by model: {target_feature_names}")
        print(f"       Features generated: {list(numerical_features.keys())}")
        return None

    return numerical_features

def predict_degradation(model, scaler, features_dict, feature_names, output_labels):
    if model is None or scaler is None or features_dict is None:
        return None 

    try:
        features_df = pd.DataFrame([features_dict], columns=feature_names)
    except Exception as e:
        print(f"ERROR creating DataFrame for prediction: {e}")
        print(f" Required columns: {feature_names}")
        print(f" Provided keys: {list(features_dict.keys())}")
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            features_scaled = scaler.transform(features_df)
    except ValueError as ve:
         print(f"ERROR scaling features: {ve}")
         if hasattr(scaler, 'n_features_in_'): print(f" Scaler expected {scaler.n_features_in_} features.")
         print(f" Input DF columns: {features_df.columns.tolist()}")
         print(f" Input DF shape: {features_df.shape}")
         return None
    except Exception as e:
        print(f"ERROR scaling features: {e}")
        return None

    try:
        predictions_scaled = model.predict(features_scaled, verbose=0)

        if predictions_scaled.ndim == 1:
            predictions_scaled = predictions_scaled.reshape(1, -1)

        results = {}
        num_outputs_model = predictions_scaled.shape[1]
        num_outputs_expected = len(output_labels)

        if num_outputs_model != num_outputs_expected:
            print(f"WARNING: Model produced {num_outputs_model} outputs, but expected {num_outputs_expected} ({output_labels}). Adjusting.")

        for i, label in enumerate(output_labels):
            if i < num_outputs_model:
                results[label] = round(predictions_scaled[0, i], 2)
            else:
                results[label] = np.nan 

        return results
    except Exception as e:
        print(f"ERROR during model prediction step: {e}")
        return None

if __name__ == "__main__":
    start_time = time.time()
    print("--- Automated Bulk City Degradation Prediction (V2) ---")
    print(f"Reading city data from: '{CITY_DATA_CSV}'")
    print(f"Loading V2 models/scalers from: '{PRETRAINED_MODEL_DIR}'")
    print(f"Using Assumptions: Usage='{USAGE_PROFILE_MAP.get(DEFAULT_USAGE_PROFILE_CHOICE,{}).get('name','INVALID')}', Start SoH={DEFAULT_INITIAL_SOH}%")
    print(f"Running for Battery Types: {BATTERY_TYPES_TO_RUN}")
    print(f"Output will be saved to: '{OUTPUT_RESULTS_CSV}'")

    try:
        cities_df_raw = pd.read_csv(CITY_DATA_CSV, dtype={'population': object})
        print(f"Loaded {len(cities_df_raw)} rows initially from CSV.")

        cities_df_raw['avg_temp_celsius'] = pd.to_numeric(cities_df_raw['avg_temp_celsius'], errors='coerce')
        cities_df_raw['population'] = pd.to_numeric(cities_df_raw['population'], errors='coerce')

        required_cols = ['city_name', 'state', 'avg_temp_celsius']
        initial_rows = len(cities_df_raw)
        cities_df_cleaned = cities_df_raw.dropna(subset=required_cols).copy()
        dropped_rows = initial_rows - len(cities_df_cleaned)
        if dropped_rows > 0:
            print(f"Dropped {dropped_rows} rows due to missing city_name, state, or avg_temp_celsius.")

        initial_rows = len(cities_df_cleaned)
        cities_df = cities_df_cleaned.drop_duplicates(subset=['city_name', 'state'], keep='first').copy()
        dropped_duplicates = initial_rows - len(cities_df)
        if dropped_duplicates > 0:
             print(f"Dropped {dropped_duplicates} duplicate city/state entries, keeping first occurrence.")

        print(f"Processing {len(cities_df)} unique city/state pairs with valid temperature.")

        if not all(col in cities_df.columns for col in ['city_name', 'state', 'avg_temp_celsius', 'population']):
            print(f"ERROR: Cleaned CSV data still missing required columns.")
            exit(1)

    except FileNotFoundError:
        print(f"ERROR: City data CSV file not found at '{CITY_DATA_CSV}'. Please create it.")
        exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read or parse CSV file '{CITY_DATA_CSV}': {e}")
        exit(1)

    loaded_models = {}
    loaded_scalers = {}
    models_loaded_ok = True
    available_model_types = list(MODEL_CONFIG.keys())
    actual_types_to_run = [] 

    print("\n--- Loading Models & Scalers ---")
    for batt_type in BATTERY_TYPES_TO_RUN:
        if batt_type not in available_model_types:
            print(f"WARN: Battery type '{batt_type}' requested but not defined in MODEL_CONFIG. Skipping.")
            continue 

        config = MODEL_CONFIG[batt_type]
        model_path = os.path.join(PRETRAINED_MODEL_DIR, config["model_file"])
        scaler_path = os.path.join(PRETRAINED_MODEL_DIR, config["scaler_file"])
        expected_features = len(config["feature_names"])

        model = load_model(model_path, config["model_type"])
        scaler = load_scaler(scaler_path, expected_features)

        if model and scaler:
            loaded_models[batt_type] = model
            loaded_scalers[batt_type] = scaler
            actual_types_to_run.append(batt_type) 
            print(f" -> Successfully loaded model and scaler for {batt_type}.")
        else:
            models_loaded_ok = False
            print(f" -> ERROR: Failed to load model or scaler for {batt_type}. This type will be skipped.")

    if not actual_types_to_run: 
        print("\nCRITICAL ERROR: No models/scalers could be loaded for ANY requested battery type. Exiting.")
        exit(1)
    elif not models_loaded_ok:
         print("\nWARNING: Failed to load models/scalers for one or more battery types. Proceeding with available types only.")

    print(f"--- Proceeding with Battery Types: {actual_types_to_run} ---")


    results_list = []
    print(f"\n--- Processing {len(cities_df)} Cities ---")
    skipped_combinations = 0
    processed_combinations = 0
    error_combinations = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(cities_df.iterrows(), total=len(cities_df), desc="Processing Cities")
    except ImportError:
        print("INFO: tqdm library not found. Install with 'pip install tqdm' for a progress bar.")
        iterator = cities_df.iterrows() 

    for index, row in iterator:
        city_name = row['city_name']
        state = row['state']
        avg_temp = row['avg_temp_celsius'] 
        population = row['population']     

        climate_choice = map_temp_to_climate_choice(avg_temp)
        traffic_choice = map_pop_to_traffic_choice(population)

        if climate_choice is None:
            skipped_combinations += len(actual_types_to_run)
            continue

        for batt_type in actual_types_to_run:
            scenario = {
                "city_name": city_name,
                "state": state,
                "battery_type": batt_type,
                "climate_choice": climate_choice,
                "usage_profile_choice": DEFAULT_USAGE_PROFILE_CHOICE,
                "traffic_choice": traffic_choice,
                "initial_soh_percent": DEFAULT_INITIAL_SOH
            }

            model = loaded_models[batt_type]
            scaler = loaded_scalers[batt_type]
            feature_names = MODEL_CONFIG[batt_type]["feature_names"]
            numerical_features = map_scenario_to_features(scenario, feature_names)

            if numerical_features is None:
                error_combinations += 1
                continue 

            predicted_results = predict_degradation(
                model, scaler, numerical_features, feature_names, OUTPUT_LABELS
            )

            result_row = {
                "City": city_name,
                "State": state,
                "BatteryType": batt_type,
                "Input_AvgTemp_C": avg_temp,
                "Input_Population": population,
                "Mapped_ClimateChoice": climate_choice,
                "Mapped_TrafficChoice": traffic_choice,
                "Assumed_UsageChoice": DEFAULT_USAGE_PROFILE_CHOICE,
                "Assumed_InitialSoH": DEFAULT_INITIAL_SOH,
                "PredictedFade_Percent": np.nan, 
                "PredictedResistanceInc_Percent": np.nan,
                "EstimatedFinalSoH_Percent": np.nan
            }
            if predicted_results:
                fade = predicted_results.get('capacity_fade_percent', np.nan)
                res_inc = predicted_results.get('resistance_increase_percent', np.nan)
                result_row["PredictedFade_Percent"] = fade
                result_row["PredictedResistanceInc_Percent"] = res_inc
                if not pd.isna(fade):
                    result_row["EstimatedFinalSoH_Percent"] = DEFAULT_INITIAL_SOH - fade
                processed_combinations += 1
            else:
                error_combinations += 1

            results_list.append(result_row)

    print("\n--- Saving Results ---")
    if results_list:
        results_df = pd.DataFrame(results_list)
        try:
            output_dir = os.path.dirname(OUTPUT_RESULTS_CSV)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            results_df.to_csv(OUTPUT_RESULTS_CSV, index=False, float_format='%.2f')
            print(f"Successfully saved {len(results_df)} prediction results (rows) to '{OUTPUT_RESULTS_CSV}'")
            print(f" -> Processed: {processed_combinations} city/battery combinations")
            print(f" -> Errors/Skipped: {error_combinations + skipped_combinations} combinations")
        except Exception as e:
            print(f"ERROR: Failed to save results CSV to '{OUTPUT_RESULTS_CSV}': {e}")
    else:
        print("No results were generated to save (check input data and model loading).")

    end_time = time.time()
    print(f"\n--- Bulk Prediction Finished ---")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")