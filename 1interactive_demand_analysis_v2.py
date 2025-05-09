import numpy as np
import pandas as pd
import joblib
import os
import warnings
import traceback
import math
import matplotlib.pyplot as plt


PRETRAINED_MODEL_DIR = "pretrained_models_v2"
OUTPUT_DIR = "interactive_demand_outputs"

MODEL_CONFIG = {
    "LiIon": {
        "model_file": "LiIon_model_v2.keras", "scaler_file": "LiIon_scaler_v2.joblib", "model_type": "NN",
        "feature_names": ["avg_temp_c", "dod", "charge_rate", "discharge_rate", "cycles_per_day", "initial_soh_percent", "traffic_level"]
    },
    "NaIon": {
        "model_file": "NaIon_model_v2.keras", "scaler_file": "NaIon_scaler_v2.joblib", "model_type": "NN",
        "feature_names": ["avg_temp_c", "dod", "charge_rate", "discharge_rate", "cycles_per_day", "initial_soh_percent", "traffic_level"]
    }
}

OUTPUT_LABELS = ["capacity_fade_percent", "resistance_increase_percent"]
TRAINING_TARGET_CYCLES = 500
FORECAST_YEARS = 15
SOH_EOL_THRESHOLD = 70.0


ASSUMED_INITIAL_RANGE_MILES = 300
ASSUMED_AVG_DAILY_MILES = { "1": 25, "2": 40, "3": 55, "4": 100 }
ASSUMED_CHARGE_TRIGGER_SOC = 20
ASSUMED_CHARGE_TARGET_SOC = 85


AVG_BATTERY_CAPACITY_KWH = 60
AVG_EV_EFFICIENCY_MILES_PER_KWH = 3.5
INITIAL_EV_FLEET_SIZE = 10000
PROJECTED_EV_FLEET_GROWTH_RATE = 0.15
PEAK_CHARGING_COINCIDENCE_FACTOR = 0.10
AVG_PEAK_CHARGER_POWER_KW = 11


PROJECTED_ANNUAL_ENERGY_SUPPLY_GROWTH = 0.10
PROJECTED_INITIAL_ANNUAL_ENERGY_SUPPLY_GWH = 50
PROJECTED_PEAK_POWER_SUPPLY_GROWTH = 0.08
PROJECTED_INITIAL_PEAK_POWER_SUPPLY_MW = 100


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
    print(f"Loading model: {model_path}", end="")
    if not os.path.exists(model_path):
        print(f"\nERROR: Model file not found at {model_path}")
        return None
    try:
        if model_type == 'NN':
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            warnings.filterwarnings('ignore', category=FutureWarning)
            try:
                import tensorflow as tf

                tf.get_logger().setLevel('ERROR')
                model = tf.keras.models.load_model(model_path, compile=False)
                print(" OK.")
                return model
            except ImportError:
                print("\nERROR: TensorFlow not found.")
                return None
            except Exception as e:
                print(f"\nERROR loading Keras model: {e}")
                return None
        else:
            model = joblib.load(model_path)
            print(" OK.")
            return model
    except Exception as e:
        print(f"\nERROR loading model file {model_path}: {e}")
        return None

def load_scaler(scaler_path):
    print(f"Loading scaler: {scaler_path}", end="")
    if not os.path.exists(scaler_path):
        print(f"\nERROR: Scaler file not found at {scaler_path}")
        return None
    try:
        scaler = joblib.load(scaler_path)
        print(" OK.")

        if not hasattr(scaler, 'transform') or not hasattr(scaler, 'mean_'):
            print(f"\nWARNING: Loaded object from {scaler_path} might not be a valid scaler?")
        return scaler
    except Exception as e:
        print(f"\nERROR loading scaler file {scaler_path}: {e}")
        return None


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
                print("Invalid number selected. Please try again.")
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
            print("Invalid selection. Please choose a valid number.")


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
            print("Invalid selection. Please choose a valid number.")


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
            print("Invalid selection. Please choose a valid number.")


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
                print("Value must be between 70 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("--- Starting Scenario Input Received ---")
    return user_choices


def map_user_input_to_base_features(user_choices, target_feature_names):
    print("Mapping user choices...", end="")
    numerical_features = {}
    climate_key=user_choices.get('climate_choice')
    usage_key=user_choices.get('usage_profile_choice')
    traffic_key=user_choices.get('traffic_choice')
    soh_val=user_choices.get('initial_soh_percent')

    if climate_key and climate_key in CLIMATE_MAP:
        numerical_features['avg_temp_c'] = CLIMATE_MAP[climate_key]['temp_c']
    else:
        print("\nERROR: Climate mapping failed.")
        return None

    if usage_key and usage_key in USAGE_PROFILE_MAP:
        params = USAGE_PROFILE_MAP[usage_key]['params']
        for feature_name in ["dod", "charge_rate", "discharge_rate", "cycles_per_day"]:
            numerical_features[feature_name] = params.get(feature_name, 0.0)
    else:
        print("\nERROR: Usage mapping failed.")
        return None

    if traffic_key and traffic_key in TRAFFIC_MAP:
        numerical_features['traffic_level'] = TRAFFIC_MAP[traffic_key]['level']
    else:
        print("\nERROR: Traffic mapping failed.")
        return None

    if soh_val is not None:
        numerical_features['initial_soh_percent'] = float(soh_val)
    else:
        print("\nERROR: SoH mapping failed.")
        return None


    if not set(target_feature_names).issubset(set(numerical_features.keys())):
        missing = list(set(target_feature_names) - set(numerical_features.keys()))
        print(f"\nERROR: Not all required features mapped. Missing: {missing}")
        return None

    print(" OK.")
    return numerical_features


def predict_single_step_fade(model, scaler, features_dict, feature_names):
    if model is None or scaler is None or features_dict is None:
        return None
    try:

        features_df = pd.DataFrame([features_dict], columns=feature_names)
    except Exception as e:
        print(f"\nERROR creating feature DataFrame: {e}")
        return None
    try:

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            features_scaled = scaler.transform(features_df)
    except Exception as e:
        print(f"\nERROR scaling features: {e}")

        print(f"Scaler expected features (count): {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'N/A'}")
        print(f"Features provided (columns): {features_df.columns.tolist()}")
        return None
    try:

        predictions = model.predict(features_scaled, verbose=0)
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)

        if "capacity_fade_percent" in OUTPUT_LABELS:
            fade_index = OUTPUT_LABELS.index("capacity_fade_percent")
            if predictions.shape[1] > fade_index:

                return predictions[0, fade_index]
            else:
                print(f"\nWARN: Prediction output shape ({predictions.shape}) doesn't contain fade index ({fade_index}).")
                return 0
        else:
            print("\nWARN: 'capacity_fade_percent' not found in configured OUTPUT_LABELS.")
            return 0
    except Exception as e:
        print(f"\nERROR during prediction step: {e}")

        return None


def run_forecast_loop_with_demand(base_features, starting_soh, usage_profile_key, model, scaler, feature_names, forecast_years, cycles_per_step, soh_eol):
    """Performs iterative forecast, calculating SoH, usable range, frequency, demand, and total miles."""
    print(f"\n--- Running Forecast & Demand Calculation ---")
    print(f"  Start SoH: {starting_soh:.1f}%, Duration: {forecast_years} years")

    if not usage_profile_key or usage_profile_key not in USAGE_PROFILE_MAP:
        print("ERROR: Invalid usage profile key provided to forecast loop.")
        return None, None, None
    cycles_per_day = USAGE_PROFILE_MAP[usage_profile_key]['params'].get('cycles_per_day', 1.0)
    if cycles_per_day <= 0:
        print("ERROR: Cycles per day must be positive.")
        return None, None, None
    avg_daily_miles = ASSUMED_AVG_DAILY_MILES.get(usage_profile_key, 40)
    print(f"  Using {cycles_per_day:.1f} cycles/day, {avg_daily_miles} avg daily miles.")

    try:
        years_per_step = cycles_per_step / (cycles_per_day * 365.25)
    except ZeroDivisionError:
        print("ERROR: Cannot calculate years per step due to zero cycles per day.")
        return None, None, None
    print(f"  Time per step: {years_per_step:.2f} years.")

    current_soh = starting_soh
    current_year = 0.0
    current_fleet_size = INITIAL_EV_FLEET_SIZE
    forecast_data = []
    time_to_eol = None
    total_lifetime_miles = None
    step = 0

    while current_year < forecast_years:
        step += 1
        start_soh_step = current_soh
        step_data = {'Year': current_year, 'SoH_Percent': start_soh_step}


        try:
            total_theoretical_range = ASSUMED_INITIAL_RANGE_MILES * (start_soh_step / 100.0)
            usable_soc_swing_percent = ASSUMED_CHARGE_TARGET_SOC - ASSUMED_CHARGE_TRIGGER_SOC
            step_data['Usable_Range_Miles'] = total_theoretical_range * (usable_soc_swing_percent / 100.0)

            if avg_daily_miles > 0 and step_data['Usable_Range_Miles'] > 0:
                step_data['Days_Between_Charges'] = step_data['Usable_Range_Miles'] / avg_daily_miles
            else:
                step_data['Days_Between_Charges'] = float('inf')

            annual_miles_per_ev = avg_daily_miles * 365.25
            if AVG_EV_EFFICIENCY_MILES_PER_KWH > 0:
                step_data['Annual_Energy_Demand_Per_EV_kWh'] = annual_miles_per_ev / AVG_EV_EFFICIENCY_MILES_PER_KWH
            else:
                 step_data['Annual_Energy_Demand_Per_EV_kWh'] = float('inf')

            step_data['Projected_Fleet_Size'] = round(current_fleet_size)
            if step_data['Annual_Energy_Demand_Per_EV_kWh'] != float('inf'):
                step_data['Total_Fleet_Annual_Energy_Demand_GWh'] = (step_data['Annual_Energy_Demand_Per_EV_kWh'] * step_data['Projected_Fleet_Size']) / 1e6
            else:
                step_data['Total_Fleet_Annual_Energy_Demand_GWh'] = float('inf')

            step_data['Est_Fleet_Peak_Power_Demand_MW'] = (step_data['Projected_Fleet_Size'] * PEAK_CHARGING_COINCIDENCE_FACTOR * AVG_PEAK_CHARGER_POWER_KW) / 1000
            step_data['Projected_Annual_Energy_Supply_GWh'] = PROJECTED_INITIAL_ANNUAL_ENERGY_SUPPLY_GWH * ((1 + PROJECTED_ANNUAL_ENERGY_SUPPLY_GROWTH) ** current_year)
            step_data['Projected_Peak_Power_Supply_MW'] = PROJECTED_INITIAL_PEAK_POWER_SUPPLY_MW * ((1 + PROJECTED_PEAK_POWER_SUPPLY_GROWTH) ** current_year)

            if step_data['Total_Fleet_Annual_Energy_Demand_GWh'] != float('inf'):
                step_data['Energy_Demand_Supply_Gap_GWh'] = step_data['Total_Fleet_Annual_Energy_Demand_GWh'] - step_data['Projected_Annual_Energy_Supply_GWh']
            else:
                step_data['Energy_Demand_Supply_Gap_GWh'] = float('inf')
            step_data['Peak_Power_Demand_Supply_Gap_MW'] = step_data['Est_Fleet_Peak_Power_Demand_MW'] - step_data['Projected_Peak_Power_Supply_MW']

        except Exception as e:
             print(f"\nERROR calculating metrics at step {step}: {e}")
             break

        forecast_data.append(step_data)


        if start_soh_step < soh_eol and time_to_eol is None:
            print(f"  INFO: SoH below {soh_eol}% at start of step {step}.")
            time_to_eol = current_year

            total_lifetime_miles = avg_daily_miles * 365.25 * time_to_eol


        step_features = base_features.copy()
        step_features['initial_soh_percent'] = start_soh_step
        predicted_fade = predict_single_step_fade(model, scaler, step_features, feature_names)

        if predicted_fade is None:
            print(f"\nERROR: Prediction failed at step {step}. Stopping forecast.")
            break
        if predicted_fade < 0:

            predicted_fade = 0.0


        current_soh -= predicted_fade
        current_year += years_per_step
        try:
            current_fleet_size *= (1 + PROJECTED_EV_FLEET_GROWTH_RATE)**years_per_step
        except OverflowError:
            print("\nWARN: Fleet size calculation overflowed. Capping growth.")
            current_fleet_size = float('inf')


        if current_year >= forecast_years:
            final_soh = max(0, current_soh)
            final_data = {'Year': current_year, 'SoH_Percent': final_soh}
            try:
                total_range = ASSUMED_INITIAL_RANGE_MILES * (final_soh / 100.0)
                final_data['Usable_Range_Miles'] = total_range * (usable_soc_swing_percent / 100.0)
                final_data['Days_Between_Charges'] = final_data['Usable_Range_Miles'] / avg_daily_miles if avg_daily_miles > 0 and final_data['Usable_Range_Miles'] > 0 else float('inf')
                final_data['Annual_Energy_Demand_Per_EV_kWh'] = (avg_daily_miles * 365.25) / AVG_EV_EFFICIENCY_MILES_PER_KWH if AVG_EV_EFFICIENCY_MILES_PER_KWH > 0 else float('inf')
                final_data['Projected_Fleet_Size'] = round(current_fleet_size) if current_fleet_size != float('inf') else float('inf')
                final_data['Total_Fleet_Annual_Energy_Demand_GWh'] = (final_data['Annual_Energy_Demand_Per_EV_kWh'] * final_data['Projected_Fleet_Size']) / 1e6 if final_data['Annual_Energy_Demand_Per_EV_kWh'] != float('inf') and final_data['Projected_Fleet_Size'] != float('inf') else float('inf')
                final_data['Est_Fleet_Peak_Power_Demand_MW'] = (final_data['Projected_Fleet_Size'] * PEAK_CHARGING_COINCIDENCE_FACTOR * AVG_PEAK_CHARGER_POWER_KW) / 1000 if final_data['Projected_Fleet_Size'] != float('inf') else float('inf')
                final_data['Projected_Annual_Energy_Supply_GWh'] = PROJECTED_INITIAL_ANNUAL_ENERGY_SUPPLY_GWH * ((1 + PROJECTED_ANNUAL_ENERGY_SUPPLY_GROWTH) ** current_year)
                final_data['Projected_Peak_Power_Supply_MW'] = PROJECTED_INITIAL_PEAK_POWER_SUPPLY_MW * ((1 + PROJECTED_PEAK_POWER_SUPPLY_GROWTH) ** current_year)
                if final_data['Total_Fleet_Annual_Energy_Demand_GWh'] != float('inf'):
                    final_data['Energy_Demand_Supply_Gap_GWh'] = final_data['Total_Fleet_Annual_Energy_Demand_GWh'] - final_data['Projected_Annual_Energy_Supply_GWh']
                else:
                    final_data['Energy_Demand_Supply_Gap_GWh'] = float('inf')
                if final_data['Est_Fleet_Peak_Power_Demand_MW'] != float('inf'):
                     final_data['Peak_Power_Demand_Supply_Gap_MW'] = final_data['Est_Fleet_Peak_Power_Demand_MW'] - final_data['Projected_Peak_Power_Supply_MW']
                else:
                     final_data['Peak_Power_Demand_Supply_Gap_MW'] = float('inf')
                forecast_data.append(final_data)
            except Exception as e:
                print(f"\nERROR calculating final metrics: {e}")



        if current_soh < soh_eol and time_to_eol is None:
            print(f"  INFO: SoH crossed {soh_eol}% threshold during step {step}.")
            if predicted_fade > 0:
                fade_fraction = max(0, min(1, (start_soh_step - soh_eol) / predicted_fade))
                time_to_eol = (current_year - years_per_step) + (fade_fraction * years_per_step)

                total_lifetime_miles = avg_daily_miles * 365.25 * time_to_eol
                print(f"    (Interpolated EoL time: {time_to_eol:.2f} years)")
            else:

                time_to_eol = current_year
                total_lifetime_miles = avg_daily_miles * 365.25 * time_to_eol



    if time_to_eol is None and current_year >= forecast_years:
        print(f"  INFO: SoH stayed above {soh_eol}% for {forecast_years} years.")
        time_to_eol = float('inf')

        total_lifetime_miles = avg_daily_miles * 365.25 * forecast_years

    print("--- Forecast Loop Finished ---")
    return forecast_data, time_to_eol, total_lifetime_miles


def plot_demand_supply_forecast(forecast_data, time_to_eol, scenario_details_str, filename="interactive_demand_supply_plot.png"):
    """Plots SoH, Usable Range, Energy Demand/Supply, Peak Power Demand/Supply."""
    if not forecast_data:
        print("No forecast results to plot.")
        return

    try:
        df = pd.DataFrame(forecast_data)
        if df.empty:
            print("Forecast data is empty, cannot plot.")
            return

        fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
        fig.suptitle(f"Forecast & Demand Analysis: {scenario_details_str}\n(Max {FORECAST_YEARS} Years, Based on {TRAINING_TARGET_CYCLES} Cycle Steps)", fontsize=14)
        ax_soh, ax_range, ax_energy, ax_power = axes[0], axes[1], axes[2], axes[3]


        ax_soh.plot(df['Year'], df['SoH_Percent'], marker='.', ls='-', label="Forecasted SoH")
        ax_soh.axhline(y=SOH_EOL_THRESHOLD, color='r', ls='--', label=f'EOL ({SOH_EOL_THRESHOLD}%)')
        if time_to_eol is not None and time_to_eol <= FORECAST_YEARS:
            ax_soh.plot(time_to_eol, SOH_EOL_THRESHOLD, marker='X', ms=10, color='r', ls='', label='_nolegend_')
            ax_soh.text(time_to_eol, SOH_EOL_THRESHOLD + 1, f'{time_to_eol:.1f} yrs EoL', color='red', ha='center', va='bottom', fontsize=9)
        ax_soh.set_ylabel("SoH (%)"); ax_soh.grid(True, ls='--'); ax_soh.legend(); ax_soh.set_ylim(bottom=max(0,SOH_EOL_THRESHOLD-15), top=105)


        ax_range.plot(df['Year'], df['Usable_Range_Miles'], marker='.', ls='-', label="Est. Usable Range", color='g')
        if time_to_eol is not None and time_to_eol <= FORECAST_YEARS:
            eol_total_range = ASSUMED_INITIAL_RANGE_MILES * (SOH_EOL_THRESHOLD / 100.0)
            eol_usable_range = eol_total_range * ((ASSUMED_CHARGE_TARGET_SOC - ASSUMED_CHARGE_TRIGGER_SOC) / 100.0)
            ax_range.plot(time_to_eol, eol_usable_range, marker='X', ms=10, color='r', ls='', label='_nolegend_')
            ax_range.text(time_to_eol, eol_usable_range + 5, f'{time_to_eol:.1f} yrs EoL', color='red', ha='center', va='bottom', fontsize=9)
        usable_soc_swing_percent = ASSUMED_CHARGE_TARGET_SOC - ASSUMED_CHARGE_TRIGGER_SOC
        initial_usable_range = ASSUMED_INITIAL_RANGE_MILES * (usable_soc_swing_percent / 100.0)
        ax_range.set_ylabel(f"Usable Range ({initial_usable_range:.0f} mi new)");
        ax_range.grid(True, ls='--'); ax_range.legend(); ax_range.set_ylim(bottom=0, top=initial_usable_range * 1.1)


        df_plot = df.replace([np.inf, -np.inf], np.nan)
        ax_energy.plot(df_plot['Year'], df_plot['Total_Fleet_Annual_Energy_Demand_GWh'], marker='.', ls='-', label="Fleet Energy Demand", color='b')
        ax_energy.plot(df_plot['Year'], df_plot['Projected_Annual_Energy_Supply_GWh'], marker='.', ls='--', label="Projected Energy Supply", color='grey')
        ax_energy.set_ylabel("Annual Energy (GWh)"); ax_energy.grid(True, ls='--'); ax_energy.legend(); ax_energy.set_ylim(bottom=0)


        ax_power.plot(df_plot['Year'], df_plot['Est_Fleet_Peak_Power_Demand_MW'], marker='.', ls='-', label="Est. Peak Power Demand", color='purple')
        ax_power.plot(df_plot['Year'], df_plot['Projected_Peak_Power_Supply_MW'], marker='.', ls='--', label="Projected Peak Power Supply", color='grey')
        ax_power.set_ylabel("Peak Power (MW)"); ax_power.grid(True, ls='--'); ax_power.legend(); ax_power.set_ylim(bottom=0)

        ax_power.set_xlabel("Years")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")
        plot_filepath = os.path.join(OUTPUT_DIR, filename)

        plt.savefig(plot_filepath)
        print(f"\n--- Forecast plot saved: {os.path.abspath(plot_filepath)} ---")

    except Exception as plot_err:
        print(f"\nERROR generating plot: {plot_err}")
        traceback.print_exc()
    finally:
         plt.close(fig)



if __name__ == "__main__":
    print("--- Interactive Battery Forecast & Demand Analysis (V2) ---")


    user_choices = get_user_input_for_forecast()
    if user_choices is None:
        print("Input cancelled or failed. Exiting.")
        exit()
    selected_battery_type = user_choices.get("battery_type")
    starting_soh = user_choices.get("initial_soh_percent")
    usage_profile_key = user_choices.get("usage_profile_choice")


    print("\n--- Loading Model & Scaler ---")
    if selected_battery_type not in MODEL_CONFIG:
        print(f"ERROR: Configuration missing for battery type '{selected_battery_type}'. Exiting.")
        exit(1)
    config = MODEL_CONFIG[selected_battery_type]
    model = load_model(os.path.join(PRETRAINED_MODEL_DIR, config["model_file"]), config["model_type"])
    scaler = load_scaler(os.path.join(PRETRAINED_MODEL_DIR, config["scaler_file"]))
    if not model or not scaler:
        print("ERROR: Failed to load model or scaler. Exiting.")
        exit(1)


    if hasattr(scaler, 'n_features_in_'):
        expected_features = len(config["feature_names"])
        if scaler.n_features_in_ != expected_features:
            print(f"\nCRITICAL ERROR: Scaler expected {scaler.n_features_in_} features, but config expects {expected_features} for {selected_battery_type}! Check files.")
            exit(1)
    else:
        print("\nWARN: Cannot verify scaler feature count. Proceeding cautiously.")
    print("--- Load OK ---")


    feature_names = config["feature_names"]
    base_features = map_user_input_to_base_features(user_choices, feature_names)
    if base_features is None:
        print("ERROR: Input mapping failed. Exiting.")
        exit(1)


    forecast_data, time_to_eol, total_lifetime_miles = run_forecast_loop_with_demand(
        base_features=base_features, starting_soh=starting_soh,
        usage_profile_key=usage_profile_key, model=model, scaler=scaler,
        feature_names=feature_names, forecast_years=FORECAST_YEARS,
        cycles_per_step=TRAINING_TARGET_CYCLES, soh_eol=SOH_EOL_THRESHOLD
    )


    if forecast_data:
        print(f"\n--- Forecast & Demand Summary for Your Scenario ---")
        results_df = pd.DataFrame(forecast_data)

        print_df = results_df.round({
             'Year': 2, 'SoH_Percent': 2, 'Usable_Range_Miles': 0, 'Days_Between_Charges': 1,
             'Annual_Energy_Demand_Per_EV_kWh': 0, 'Projected_Fleet_Size': 0,
             'Total_Fleet_Annual_Energy_Demand_GWh': 1, 'Est_Fleet_Peak_Power_Demand_MW': 1,
             'Projected_Annual_Energy_Supply_GWh': 1, 'Projected_Peak_Power_Supply_MW': 1,
             'Energy_Demand_Supply_Gap_GWh': 1, 'Peak_Power_Demand_Supply_Gap_MW': 1
        })

        print_df = print_df.replace([np.inf, -np.inf], "N/A")


        print_df_renamed = print_df.rename(columns={
            'Usable_Range_Miles': 'UsableRange(mi)',
            'Total_Fleet_Annual_Energy_Demand_GWh': 'FleetDemand(GWh)',
            'Energy_Demand_Supply_Gap_GWh': 'EnergyGap(GWh)',
            'Est_Fleet_Peak_Power_Demand_MW': 'FleetPeakPwr(MW)',
            'Peak_Power_Demand_Supply_Gap_MW': 'PowerGap(MW)',
            'Days_Between_Charges': 'ChargeEvery(days)'
        })

        cols_to_print = ['Year', 'SoH_Percent', 'UsableRange(mi)', 'ChargeEvery(days)',
                         'FleetDemand(GWh)', 'EnergyGap(GWh)',
                         'FleetPeakPwr(MW)', 'PowerGap(MW)']

        if all(col in print_df_renamed.columns for col in cols_to_print):
             print(print_df_renamed[cols_to_print].to_string(index=False))
        else:
             print("ERROR: One or more columns missing for summary table.")
             print(print_df_renamed.head())

        print("-" * 70)
        if time_to_eol is not None and time_to_eol <= FORECAST_YEARS:
            print(f"  Estimated Time to Reach {SOH_EOL_THRESHOLD}% SoH: {time_to_eol:.2f} years")
            if total_lifetime_miles is not None:
                print(f"  Estimated Total Lifetime Mileage: {total_lifetime_miles:,.0f} miles")
        elif time_to_eol == float('inf'):
             print(f"  SoH remained above {SOH_EOL_THRESHOLD}% for {FORECAST_YEARS} years.")
             if total_lifetime_miles is not None:
                  print(f"  Estimated Mileage over {FORECAST_YEARS} years: {total_lifetime_miles:,.0f} miles")
        else:
             print("  Time to EoL could not be determined.")

        print("-" * 70)
        print("NOTE: Supply figures and Peak Power Demand are based on *projected* growth assumptions.")
        print("      A positive Gap indicates Demand potentially exceeds Projected Supply.")
        print("      Real infrastructure data (Step 4) is needed for valid comparison.")


        batt_str = selected_battery_type
        clim_str = CLIMATE_MAP.get(user_choices.get("climate_choice"),{}).get("name","?")
        use_str = USAGE_PROFILE_MAP.get(usage_profile_key,{}).get("name","?")
        traf_str = TRAFFIC_MAP.get(user_choices.get("traffic_choice"),{}).get("name","?")
        scenario_details = f"{batt_str} | Start {starting_soh:.0f}% | {clim_str} | {use_str} | {traf_str}"

        safe_scenario_name = f"demand_forecast_{batt_str}_start{starting_soh:.0f}soh_{clim_str}_{use_str}_{traf_str}"
        safe_scenario_name = safe_scenario_name.replace(' ','_').replace('/','-').replace('(','').replace(')','').replace('<','lt').replace('>','gt').replace('%','')
        plot_filename = f"{safe_scenario_name}.png"

        plot_demand_supply_forecast(forecast_data, time_to_eol, scenario_details, filename=plot_filename)


        csv_filename = os.path.join(OUTPUT_DIR, f"detailed_{safe_scenario_name}.csv")
        try:

            results_df['TimeToEOL_Years'] = time_to_eol if time_to_eol is not None else np.nan
            results_df['Total_Lifetime_Miles'] = total_lifetime_miles if total_lifetime_miles is not None else np.nan
            results_df.to_csv(csv_filename, index=False, float_format='%.3f')
            print(f"--- Detailed forecast data saved: {os.path.abspath(csv_filename)} ---")
        except Exception as e:
            print(f"ERROR saving detailed CSV: {e}")
            traceback.print_exc()
    else:
        print("\nERROR: Forecast could not be completed.")

    print(f"\n{'='*60}")
    print("--- Interactive Demand Analysis Finished ---")