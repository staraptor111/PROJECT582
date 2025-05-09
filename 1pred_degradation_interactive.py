
import numpy as np
import pandas as pd
import joblib
import os
import warnings
import traceback

PRETRAINED_MODEL_DIR = "pretrained_models"

MODEL_CONFIG = {
    "LiIon": {
        "model_file": "LiIon_model.keras", 
        "scaler_file": "LiIon_scaler.joblib",
        "model_type": "NN", 
        "feature_names": [
            "avg_temp_c", "dod", "charge_rate", "discharge_rate", "cycles_per_day"
        ]
    },
    "NaIon": {
        "model_file": "NaIon_model.keras", 
        "scaler_file": "NaIon_scaler.joblib",
        "model_type": "NN", 
        "feature_names": [
            "avg_temp_c", "dod", "charge_rate", "discharge_rate", "cycles_per_day"
        ]
    }
}

OUTPUT_LABELS = ["capacity_fade_percent", "resistance_increase_percent"]
TRAINING_TARGET_CYCLES = 500 

def load_model(model_path, model_type):
    print(f"Attempting to load model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return None
    try:
        if model_type == 'NN':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
            warnings.filterwarnings('ignore', category=FutureWarning)
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path, compile=False) 
                print("Keras model loaded successfully.")
                return model
            except ImportError:
                print("ERROR: TensorFlow not found, cannot load Keras model.")
                return None
            except Exception as e:
                print(f"ERROR loading Keras model: {e}")
                traceback.print_exc()
                return None
        else: 
            model = joblib.load(model_path)
            print("Joblib model loaded successfully.")
            return model
    except Exception as e:
        print(f"ERROR loading model file {model_path}: {e}")
        traceback.print_exc()
        return None

def load_scaler(scaler_path):
    print(f"Attempting to load scaler from: {scaler_path}")
    if not os.path.exists(scaler_path):
        print(f"ERROR: Scaler file not found at {scaler_path}")
        return None
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
        if not hasattr(scaler, 'transform') or not hasattr(scaler, 'mean_'):
             print(f"WARNING: Loaded object from {scaler_path} might not be a valid scaler.")
        return scaler
    except Exception as e:
        print(f"ERROR loading scaler file {scaler_path}: {e}")
        traceback.print_exc()
        return None

def get_user_input_simple(feature_names_example):
    print("\n--- Enter Scenario Details ---")
    user_inputs = {}

    available_types = list(MODEL_CONFIG.keys())
    while True:
        print(f"Available battery types: {', '.join(available_types)}")
        batt_type = input("Select battery type: ").strip()
        if batt_type in available_types:
            user_inputs['battery_type'] = batt_type
            break
        else:
            print("Invalid selection. Please choose from the available types.")

    print("\nClimate Conditions:")
    while True:
        try:
            temp_str = input(f"Enter average ambient temperature (°C) [Feature: avg_temp_c]: ")
            user_inputs['avg_temp_c'] = float(temp_str)
            break
        except ValueError:
            print("Invalid input. Please enter a number for temperature.")

    print("\nVehicle and Usage:")
    while True:
        v_type = input("Enter vehicle type (EV / Hybrid) [Not a direct model feature, informational]: ").strip().upper()
        if v_type in ["EV", "HYBRID"]:
            user_inputs['vehicle_type'] = v_type 
            break
        else:
            print("Invalid input. Please enter EV or Hybrid.")

    while True:
        try:
            dod_str = input(f"Enter typical Depth of Discharge (DoD) per cycle (0.0 to 1.0) [Feature: dod]: ")
            dod = float(dod_str)
            if 0.0 < dod <= 1.0:
                user_inputs['dod'] = dod
                break
            else:
                 print("DoD must be between 0.0 (exclusive) and 1.0 (inclusive).")
        except ValueError:
            print("Invalid input. Please enter a number for DoD.")

    while True:
        try:
            cr_str = input(f"Enter typical Charge Rate (C-rate) [Feature: charge_rate]: ")
            user_inputs['charge_rate'] = float(cr_str)
            if user_inputs['charge_rate'] > 0: break
            else: print("Charge rate must be positive.")
        except ValueError:
            print("Invalid input. Please enter a number for charge rate.")

    while True:
        try:
            dr_str = input(f"Enter typical Discharge Rate (C-rate) during driving [Feature: discharge_rate]: ")
            user_inputs['discharge_rate'] = float(dr_str)
            if user_inputs['discharge_rate'] > 0: break
            else: print("Discharge rate must be positive.")
        except ValueError:
            print("Invalid input. Please enter a number for discharge rate.")

    while True:
        try:
            cpd_str = input(f"Enter average number of full equivalent cycles per day [Feature: cycles_per_day]: ")
            user_inputs['cycles_per_day'] = float(cpd_str)
            if user_inputs['cycles_per_day'] > 0: break
            else: print("Cycles per day must be positive.")

        except ValueError:
            print("Invalid input. Please enter a number for cycles per day.")

    print("--- Input Received ---")
    return user_inputs


def map_inputs_to_features(user_inputs, feature_names):
    print("Mapping inputs to model features...")
    features = {}
    missing_features = []

    for feature in feature_names:
        if feature in user_inputs:
            try:
                features[feature] = float(user_inputs[feature])
            except (ValueError, TypeError):
                 print(f"ERROR: Could not convert input for '{feature}' ('{user_inputs[feature]}') to a number.")
                 missing_features.append(feature)
        else:
            print(f"ERROR: Critical input feature '{feature}' is missing from user inputs.")
            missing_features.append(feature)

    if missing_features:
        print(f"Cannot proceed due to missing or invalid features: {', '.join(missing_features)}")
        return None

    print("Input mapping complete.")
    return features


def predict_degradation(model, scaler, features_dict, feature_names, output_labels):
    if model is None or scaler is None or features_dict is None:
        print("ERROR: Model, scaler, or features dictionary is missing. Cannot predict.")
        return None

    print("\n--- Predicting Degradation ---")
    try:
        features_df = pd.DataFrame([features_dict], columns=feature_names)
        print("Input DataFrame for prediction:")
        print(features_df)
    except Exception as e:
        print(f"ERROR creating DataFrame from features: {e}")
        print(f"Feature names required: {feature_names}")
        print(f"Features provided: {list(features_dict.keys())}") 
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning) 
            features_scaled = scaler.transform(features_df)
        print("Scaled features (first row):", features_scaled[0])
    except Exception as e:
        print(f"ERROR scaling features: {e}")
        if hasattr(scaler, 'feature_names_in_'):
             print("Scaler expected features:", scaler.feature_names_in_)
        elif hasattr(scaler, 'n_features_in_'):
             print("Scaler expected number of features:", scaler.n_features_in_)
        print("Actual features provided to scaler:", features_df.columns.tolist())
        traceback.print_exc()
        return None

    try:
        predictions_scaled = model.predict(features_scaled)
        if predictions_scaled.ndim == 1:
             predictions_scaled = predictions_scaled.reshape(1, -1) 

        print("Raw model predictions:", predictions_scaled[0])

        num_outputs = predictions_scaled.shape[1]
        results = {label: round(predictions_scaled[0, i], 2)
                   for i, label in enumerate(output_labels) if i < num_outputs}

        if len(results) != len(output_labels):
             print(f"WARNING: Model produced {num_outputs} outputs, but expected {len(output_labels)}. Displaying available results.")

        return results

    except Exception as e:
        print(f"ERROR during model prediction: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("--- Interactive Battery Degradation Predictor ---")
    print(f"Loading models and scalers from: '{PRETRAINED_MODEL_DIR}'")
    print(f"Prediction assumes model was trained for approx. {TRAINING_TARGET_CYCLES} cycles.")

    example_feature_names = list(MODEL_CONFIG.values())[0]['feature_names'] if MODEL_CONFIG else []

    user_scenario = get_user_input_simple(example_feature_names)

    if user_scenario is None:
        print("Exiting due to input error or cancellation.")
        exit(1)

    selected_battery_type = user_scenario.get('battery_type')
    if not selected_battery_type or selected_battery_type not in MODEL_CONFIG:
        print(f"ERROR: Invalid or missing battery type in user input: {selected_battery_type}")
        exit(1)

    config = MODEL_CONFIG[selected_battery_type]
    model_path = os.path.join(PRETRAINED_MODEL_DIR, config["model_file"])
    scaler_path = os.path.join(PRETRAINED_MODEL_DIR, config["scaler_file"])

    model = load_model(model_path, config["model_type"])
    scaler = load_scaler(scaler_path)

    if model is None or scaler is None:
        print("\nERROR: Failed to load necessary model or scaler.")
        print("Please ensure the following files exist in the 'pretrained_models' directory:")
        print(f"  - {config['model_file']}")
        print(f"  - {config['scaler_file']}")
        print("These files should be generated by running the main training script")
        print("in Train/Test mode (USE_CROSS_VALIDATION = False) and copying")
        print("the resulting *simply named* files (without timestamp) here.")
        exit(1)

    feature_names_for_model = config["feature_names"]
    numerical_features = map_inputs_to_features(user_scenario, feature_names_for_model)

    if numerical_features is None:
        print("Exiting due to input mapping error.")
        exit(1)

    predicted_results = predict_degradation(model, scaler, numerical_features,
                                             feature_names_for_model, OUTPUT_LABELS)

    print("\n--- Prediction Results ---")
    if predicted_results:
        print(f"Scenario Details:")
        for key, val in user_scenario.items():
             print(f"  - {key.replace('_', ' ').title()}: {val}")


        print(f"\nPredicted Degradation (after approx. {TRAINING_TARGET_CYCLES} cycles):")
        for label, value in predicted_results.items():
            unit = "%" 
            print(f"  - {label.replace('_', ' ').title()}: {value:.2f} {unit}")

        print("\nNote: These predictions are based on the simplified input mapping")
        print("and the pre-trained surrogate model. Actual degradation may vary.")
    else:
        print("Prediction failed.")

    print("\n--- End of Prediction ---")