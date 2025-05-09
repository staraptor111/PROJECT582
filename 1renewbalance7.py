import pybamm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
import time as timer
from pathlib import Path
import sys
import traceback 
from collections.abc import Iterable 
import functools 
import json 
import inspect 
import warnings 

try:
    from fuzzywuzzy import process, fuzz
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

CONFIG = {
    "MODEL_DIR": Path("C:/Users/snsur/Desktop/ster/"),
    "MODEL_FILENAME": "my_actual_degradation_model.py",
    "MODEL_FUNCTION_NAME": "my_actual_degradation_model",
    "CITIES_CSV_PATH": Path("C:/Users/snsur/Desktop/ster/city_temperatures_clean.csv"),
    "CITY_RENEWABLE_CAPACITY_PATH": Path(r"C:\Users\snsur\Desktop\ster\cleaned_city_renewable_supply_solar_wind (2).csv"),
    "RENEWABLE_PROFILES_DATA_PATH": Path(r"C:\Users\snsur\Desktop\ster\cleaned_renewable_profiles (1) (1).csv"),
    "PYBAMM_LOOKUP_PATH": Path("./pybamm_temperature_lookup.csv"),
    "SUMMARY_OUTPUT_FILENAME": "multi_city_simulation_summary_v3.6_combined_renew.csv",
    "HOURLY_RESULTS_DIR": Path("./results_condensed_v3.6_combined_renew"),

    "CITY_COL": 'city_name', "TEMP_COL": 'avg_temp_celsius', "POP_COL": 'population',
    "RENEW_CITY_COL": 'city_name',
    "RENEW_SOLAR_PROFILE_ID_COL": 'solar_profile_id', 
    "RENEW_WIND_PROFILE_ID_COL": 'wind_profile_id',   
    "RENEW_MW_YEAR_COLS_TUPLES": [
        ('mw_solar_2025', 'mw_wind_2025'),
        ('mw_solar_2030', 'mw_wind_2030'),
        ('mw_solar_2035', 'mw_wind_2035'),
    ],
    "PROFILE_ID_COL": 'profile_id', "PROFILE_HOUR_COL": 'hour_of_year', "PROFILE_VALUE_COL": 'kw_per_mw',

    "FORECAST_YEARS": 10,
    "EV_ADOPTION_RATE_PER_CAPITA": 0.01,
    "MIN_INITIAL_FLEET_SIZE": 10,
    "FLEET_GROWTH_RATE": 0.15,

    "AVG_DAILY_KM_PER_EV": 40,
    "BASE_EFFICIENCY_KWH_KM": 0.18,
    "CYCLES_PER_DAY": 1.0,
    "INITIAL_AVG_SOH_PERCENT": 100.0,
    "INITIAL_AVG_RESISTANCE_PERCENT": 100.0,
    "BATTERY_NOMINAL_CAPACITY_KWH": 60,
    "CHEMISTRY": "Li-ion",
    "TRAFFIC_LEVEL": 1,
    "USER_PROFILE": "Average User",

    "PYBAMM_PARAM_SET": "Chen2020",
    "PYBAMM_SIM_TYPE": 'charge',
    "PYBAMM_SIM_CRATE": 1.5,
    "HEAT_TRANSFER_COEFF": 25,
    "PYBAMM_INITIAL_SOC": 0.1,

    "CHARGER_MIX": {'L2':{'power':7.0,'fraction':0.60},'DCFC_50':{'power':50.0,'fraction':0.25},'DCFC_150':{'power':150.0,'fraction':0.15}},
    "RESISTANCE_IMPACT_FACTOR": 0.2,
    "CHARGE_START_HOUR": 0,
    "CHARGE_END_HOUR": 4,
    "SMART_CHARGING_STRATEGY": 'unmanaged',

    "USE_GRID_STORAGE": True,
    "STORAGE_CAPACITY_KWH": 20000,
    "STORAGE_MAX_CHARGE_KW": 5000,
    "STORAGE_MAX_DISCHARGE_KW": 5000,
    "STORAGE_EFFICIENCY": 0.85,

    "INITIAL_MAX_GRID_POWER_KW": 50000,
    "GRID_LIMIT_GROWTH_RATE": 0.10,

    "CAPEX_SOLAR_USD_PER_KW": 1050,
    "CAPEX_WIND_USD_PER_KW": 1600,
    "CAPEX_STORAGE_USD_PER_KWH": 250,
    "CAPEX_STORAGE_USD_PER_KW": 180,
    "OPEX_SOLAR_PERCENT_CAPEX": 1.5,
    "OPEX_WIND_PERCENT_CAPEX": 2.5,
    "OPEX_STORAGE_PERCENT_CAPEX": 2.0,
    "GRID_PRICE_USD_PER_KWH": 0.16,
    "DISCOUNT_RATE": 0.07,
    "ANALYSIS_LIFETIME_YEARS": 25,
}

try:
    CONFIG['RENEW_CAPACITY_YEARS'] = sorted([int(cols[0].split('_')[-1]) for cols in CONFIG['RENEW_MW_YEAR_COLS_TUPLES'] if cols[0].startswith('mw_solar_') and cols[0].split('_')[-1].isdigit()])
    if not CONFIG['RENEW_CAPACITY_YEARS']: raise ValueError("No valid years parsed from RENEW_MW_YEAR_COLS_TUPLES (solar cols).")
    CONFIG['SIM_START_YEAR'] = CONFIG['RENEW_CAPACITY_YEARS'][0]
    CONFIG['RENEW_SOLAR_MW_YEAR_COLS'] = [cols[0] for cols in CONFIG['RENEW_MW_YEAR_COLS_TUPLES']]
    CONFIG['RENEW_WIND_MW_YEAR_COLS'] = [cols[1] for cols in CONFIG['RENEW_MW_YEAR_COLS_TUPLES']]
    CONFIG['ALL_RENEW_MW_COLS'] = CONFIG['RENEW_SOLAR_MW_YEAR_COLS'] + CONFIG['RENEW_WIND_MW_YEAR_COLS']
except Exception as e: print(f"ERROR [Config]: Invalid RENEW_MW_YEAR_COLS_TUPLES structure or naming: {CONFIG['RENEW_MW_YEAR_COLS_TUPLES']}. {e}"); sys.exit(1)

try: import pyarrow; PYARROW_AVAILABLE = True
except ImportError: print("Warning: 'pyarrow' not found. Parquet saving will be skipped."); PYARROW_AVAILABLE = False

cities_df_global = None
city_renewable_capacity_df_global = None
renewable_profiles_data_df_global = None
_pybamm_lookup_table = None

def placeholder_degradation_model(initial_soh_percent, resistance_initial_percent, num_cycles, avg_temperature_c, peak_charge_temperature_c, traffic_condition, usage_pattern, cell_chemistry, **kwargs):
    if np.isnan(avg_temperature_c) or np.isnan(peak_charge_temperature_c): return 0.0, 0.0
    b_fade = 0.00015 if cell_chemistry == "Li-ion" else 0.00020
    b_res = 0.00025 if cell_chemistry == "Li-ion" else 0.00020
    t_fact = 1 + (max(0, avg_temperature_c - 25) / 15)**1.5 
    try: traffic_numeric = float(traffic_condition)
    except (ValueError, TypeError): traffic_numeric = 1.0 
    tr_fact = 1 + (traffic_numeric * 0.1) 
    u_fact = 1.5 if usage_pattern == "Taxi/Ride Share" else (0.8 if usage_pattern == "Gentle Commuter" else 1.0) 
    s_cap_fact = max(0.5, initial_soh_percent / 100.0) 
    cap_d_percent = num_cycles * b_fade * t_fact * tr_fact * u_fact * s_cap_fact
    res_d_percent = num_cycles * b_res * t_fact * tr_fact * u_fact
    final_cap_d = min(cap_d_percent, initial_soh_percent) 
    final_res_d = max(0, res_d_percent) 
    return final_cap_d, final_res_d

def load_custom_degradation_model(model_dir, model_filename, function_name):
    if not model_dir.is_dir():
        return None, f"Directory not found: {model_dir}"
    model_dir_str = str(model_dir.resolve())
    if model_dir_str not in sys.path:
        sys.path.append(model_dir_str)
    try:
        module_name = model_filename.replace(".py", "")
        degradation_module = __import__(module_name) 
        imported_func = getattr(degradation_module, function_name)
        sig = inspect.signature(imported_func)
        required_params = {'initial_soh_percent', 'resistance_initial_percent', 'num_cycles',
                           'avg_temperature_c', 'peak_charge_temperature_c', 'traffic_condition',
                           'usage_pattern', 'cell_chemistry'}
        if required_params.issubset(sig.parameters.keys()):
            print(f"INFO [DegModel]: Imported custom model: '{function_name}'.")
            return imported_func, None
        else:
            missing = required_params - set(sig.parameters.keys())
            msg = f"Imported function '{function_name}' missing parameters: {missing}."
    except ImportError as e:
        msg = f"Failed importing model file ('{model_filename}'): {e}."
    except AttributeError:
        msg = f"Function '{function_name}' not found in '{model_filename}'."
    except Exception as e:
        msg = f"Unexpected error during custom model import: {e}"
        traceback.print_exc()
    print(f"WARN [DegModel]: {msg}")
    return None, msg

predict_degradation_wrapper, load_error = load_custom_degradation_model(CONFIG['MODEL_DIR'], CONFIG['MODEL_FILENAME'], CONFIG['MODEL_FUNCTION_NAME'])
if predict_degradation_wrapper is None:
    print(f"INFO [DegModel]: Using placeholder degradation model. Reason: {load_error}")
    predict_degradation_wrapper = placeholder_degradation_model

def load_pybamm_lookup():
    global _pybamm_lookup_table
    if _pybamm_lookup_table is not None:
        return _pybamm_lookup_table
    lookup_path = CONFIG['PYBAMM_LOOKUP_PATH']
    if not lookup_path.is_file():
        raise FileNotFoundError(f"PyBaMM lookup file not found: {lookup_path}")
    try:
        print(f"INFO [PyBaMM]: Loading lookup table: {lookup_path}")
        df = pd.read_csv(lookup_path)
        required_cols = ['ambient_temp_c', 'avg_batt_temp_c', 'peak_batt_temp_c']
        assert all(col in df.columns for col in required_cols), f"Lookup table missing required columns: {required_cols}"
        df = df.sort_values(by="ambient_temp_c").reset_index(drop=True)
        if df[required_cols].isnull().values.any():
            print("WARN [PyBaMM]: NaNs found in lookup table! Interpolation might yield NaNs.")
        _pybamm_lookup_table = df
        print("INFO [PyBaMM]: Lookup table loaded successfully.")
        return _pybamm_lookup_table
    except Exception as e:
        print(f"ERROR [PyBaMM]: Failed loading lookup table '{lookup_path}': {e}")
        _pybamm_lookup_table = pd.DataFrame() 
        raise RuntimeError("PyBaMM lookup failed.") from e

def run_pybamm_simulation(ambient_temp_c, **kwargs):
    if np.isnan(ambient_temp_c):
        return np.nan, np.nan
    try:
        lookup_df = load_pybamm_lookup()
        if lookup_df.empty:
            print("ERROR [PyBaMM]: Lookup table is empty, cannot interpolate.")
            return np.nan, np.nan

        x_lookup = lookup_df['ambient_temp_c']
        y_avg = lookup_df['avg_batt_temp_c']
        y_peak = lookup_df['peak_batt_temp_c']

        interp_avg_t = np.interp(ambient_temp_c, x_lookup, y_avg)
        interp_peak_t = np.interp(ambient_temp_c, x_lookup, y_peak)

        min_t, max_t = x_lookup.min(), x_lookup.max()
        if not (min_t <= ambient_temp_c <= max_t):
            print(f"WARN [PyBaMM]: Ambient temp {ambient_temp_c:.1f}C outside pre-computed range [{min_t:.1f}C, {max_t:.1f}C]. Result is extrapolated.")

        if np.isnan(interp_avg_t) or np.isnan(interp_peak_t):
             print(f"ERROR [PyBaMM]: Interpolation resulted in NaN for ambient temp {ambient_temp_c:.1f}C. Check lookup table data.")
             return np.nan, np.nan

        return interp_avg_t, interp_peak_t
    except Exception as e:
        print(f"ERROR [PyBaMM]: Lookup/interpolation failed for ambient temp {ambient_temp_c:.1f}C: {e}")
        return np.nan, np.nan

def forecast_fleet_degradation(years, initial_soh_cap, initial_soh_res_percent, cycles_per_day,
                               avg_temp_c, peak_temp_c, traffic_level, user_profile, chemistry, config):
    if np.isnan(avg_temp_c) or np.isnan(peak_temp_c):
        print("ERROR [DegForecast]: NaN temperature input.")
        return pd.DataFrame(), "NaN temp input"

    if predict_degradation_wrapper is None:
        print("ERROR [DegForecast]: Degradation model function not available.")
        return pd.DataFrame(), "Degradation model not available"

    history = []
    current_soh_cap = float(initial_soh_cap)
    current_res_pct = float(initial_soh_res_percent)
    cycles_per_year = float(cycles_per_day * 365.25)
    initial_resistance_pct_config = config.get('INITIAL_AVG_RESISTANCE_PERCENT', 100.0)

    history.append({"Year": 0, "Avg_SoH_Cap_Percent": current_soh_cap, "Resistance_Percent": current_res_pct})
    error_msg = None
    EOL_THRESHOLD = 20.0 

    for year in range(1, years + 1):
        if current_soh_cap <= EOL_THRESHOLD:
            print(f"INFO [DegForecast]: Reached EOL threshold ({EOL_THRESHOLD}%) in Year {year-1}. Stopping forecast.")
            break

        try:
            fade_delta_pp, res_increase_pp = predict_degradation_wrapper(
                initial_soh_percent=current_soh_cap,
                resistance_initial_percent=current_res_pct, 
                num_cycles=cycles_per_year,
                avg_temperature_c=avg_temp_c,
                peak_charge_temperature_c=peak_temp_c,
                traffic_condition=traffic_level,
                usage_pattern=user_profile,
                cell_chemistry=chemistry
            )

            if not isinstance(fade_delta_pp, (int, float)) or not isinstance(res_increase_pp, (int, float)) \
               or np.isnan(fade_delta_pp) or np.isnan(res_increase_pp) \
               or fade_delta_pp < 0 or res_increase_pp < 0:
                print(f"WARN [DegForecast]: Invalid degradation prediction (NaN, negative, or non-numeric) in Year {year}: Fade={fade_delta_pp}, Res={res_increase_pp}. Keeping state constant for this year.")
                error_msg = f"Invalid prediction Year {year}"
                history.append({"Year": year, "Avg_SoH_Cap_Percent": current_soh_cap, "Resistance_Percent": current_res_pct})
                continue 

            current_soh_cap -= fade_delta_pp
            current_res_pct += res_increase_pp

            current_soh_cap = max(0.0, current_soh_cap)
            current_res_pct = max(initial_resistance_pct_config, current_res_pct) 

            history.append({"Year": year, "Avg_SoH_Cap_Percent": current_soh_cap, "Resistance_Percent": current_res_pct})

        except Exception as e:
            print(f"ERROR [DegForecast]: Exception during prediction in Year {year}: {e}")
            traceback.print_exc()
            error_msg = f"Exception Year {year}: {e}"
            break 

    last_entry_year = history[-1]["Year"]
    if last_entry_year < years:
        last_soh = history[-1]["Avg_SoH_Cap_Percent"]
        last_res = history[-1]["Resistance_Percent"]
        for future_year in range(last_entry_year + 1, years + 1):
            history.append({"Year": future_year, "Avg_SoH_Cap_Percent": last_soh, "Resistance_Percent": last_res})

    return pd.DataFrame(history), error_msg

def get_city_financial_factors(city_name, config):
    print(f"INFO [FinancialFactors]: Getting factors for {city_name} (using PLACEHOLDER defaults). Modify this function to load real data.")
    return {
        'grid_price_usd_kwh': config['GRID_PRICE_USD_PER_KWH'], 
        'capex_mult': 1.0, 
        'opex_mult': 1.0,  
    }

def _generate_single_renewable_series(capacity_points_dict, profile_id, renewable_profiles_df, config):
    profile_id_col = config['PROFILE_ID_COL']
    profile_hour_col = config['PROFILE_HOUR_COL']
    profile_value_col = config['PROFILE_VALUE_COL']
    forecast_years = config['FORECAST_YEARS']
    sim_start_year = config['SIM_START_YEAR']
    sim_start_date = pd.Timestamp(f'{sim_start_year}-01-01')
    target_hours = 8760 

    if not capacity_points_dict or not profile_id:
        sim_end_date = sim_start_date + pd.DateOffset(years=forecast_years, days=-1)
        full_end_date = sim_end_date.replace(hour=23, minute=59, second=59)
        hourly_index = pd.date_range(start=sim_start_date, end=full_end_date, freq='h')
        return pd.Series(0.0, index=hourly_index), 0.0, 0.0 

    try:
        years_in_data = sorted(capacity_points_dict.keys())
        mw_values_in_data = [capacity_points_dict[y] for y in years_in_data]
        mw_values_in_data = [max(0.0, float(mw)) if pd.notna(mw) else 0.0 for mw in mw_values_in_data]

        initial_capacity_mw = float(np.interp(sim_start_year, years_in_data, mw_values_in_data, left=mw_values_in_data[0], right=mw_values_in_data[-1])) if years_in_data else 0.0
        initial_capacity_mw = max(0.0, initial_capacity_mw)

        growth_rate = 0.0
        if len(years_in_data) >= 2:
            first_year, last_year = years_in_data[0], years_in_data[-1]
            first_mw, last_mw = mw_values_in_data[0], mw_values_in_data[-1]
            total_years = last_year - first_year
            if total_years > 0 and first_mw > 1e-6: growth_rate = ((last_mw / first_mw)**(1.0 / total_years)) - 1
            elif first_mw <= 1e-6 and last_mw > 1e-6: growth_rate = 0.10
        growth_rate = np.clip(growth_rate, -0.5, 2.0)
    except Exception as e: print(f"ERROR [_genSingleRenew]: Failed processing capacity for profile '{profile_id}': {e}"); raise

    try:
        hourly_profile_data = renewable_profiles_df[renewable_profiles_df[profile_id_col] == profile_id].copy()
        if hourly_profile_data.empty:
            print(f"WARN [_genSingleRenew]: Profile ID '{profile_id}' not found in profiles data. Returning zero series.")
            sim_end_date = sim_start_date + pd.DateOffset(years=forecast_years, days=-1); full_end_date = sim_end_date.replace(hour=23, minute=59, second=59)
            hourly_index = pd.date_range(start=sim_start_date, end=full_end_date, freq='h'); return pd.Series(0.0, index=hourly_index), initial_capacity_mw, growth_rate

        assert profile_hour_col in hourly_profile_data and profile_value_col in hourly_profile_data, f"Profile data missing cols for '{profile_id}'."
        hourly_profile_data.dropna(subset=[profile_hour_col, profile_value_col], inplace=True)
        assert not hourly_profile_data.empty, f"Profile ID '{profile_id}' has no valid numeric data."
        hourly_profile_data.sort_values(by=profile_hour_col, inplace=True)
        hourly_shape_values = hourly_profile_data[profile_value_col].values
        max_raw_value = np.max(hourly_shape_values) if len(hourly_shape_values) > 0 else 0
        hourly_shape_cf = hourly_shape_values / 1000.0 if max_raw_value > 1.1 else hourly_shape_values
        hourly_shape_cf = np.clip(np.nan_to_num(hourly_shape_cf, nan=0.0), 0.0, 1.0)

        if len(hourly_shape_cf) != target_hours:
            temp_index = pd.date_range(start='2023-01-01', freq='h', periods=len(hourly_shape_cf))
            target_index = pd.date_range(start='2023-01-01', periods=target_hours, freq='h')
            profile_resampled = pd.Series(hourly_shape_cf, index=temp_index).reindex(target_index).interpolate(method='linear').ffill().bfill().fillna(0)
            hourly_shape_cf = profile_resampled.values; assert len(hourly_shape_cf) == target_hours, f"Resampling failed for profile '{profile_id}'."
    except Exception as e: print(f"ERROR [_genSingleRenew]: Failed processing profile '{profile_id}': {e}"); raise

    try:
        sim_end_date = sim_start_date + pd.DateOffset(years=forecast_years, days=-1); full_end_date = sim_end_date.replace(hour=23, minute=59, second=59)
        hourly_index = pd.date_range(start=sim_start_date, end=full_end_date, freq='h')
        sim_years_array = np.arange(sim_start_year, sim_start_year + forecast_years)
        annual_capacities_mw = np.interp(sim_years_array, years_in_data, mw_values_in_data, left=mw_values_in_data[0], right=mw_values_in_data[-1]) if years_in_data else np.zeros_like(sim_years_array)
        annual_capacities_mw = np.maximum(0.0, annual_capacities_mw)
        annual_capacity_map_mw = dict(zip(sim_years_array, annual_capacities_mw))

        day_of_year = hourly_index.dayofyear; hour_of_day = hourly_index.hour
        effective_day_of_year = np.where(day_of_year > 365, 365, day_of_year)
        hour_of_year_indices = np.clip((effective_day_of_year - 1) * 24 + hour_of_day, 0, target_hours - 1).astype(int)
        capacity_factor_arr = hourly_shape_cf[hour_of_year_indices]
        current_sim_year_arr = hourly_index.year.values
        current_capacity_mw_arr = np.array([annual_capacity_map_mw.get(yr, initial_capacity_mw) for yr in current_sim_year_arr]).astype(float)
        supply_kw_values = capacity_factor_arr * current_capacity_mw_arr * 1000.0
        supply_kw_series = pd.Series(supply_kw_values, index=hourly_index).fillna(0.0)
        return supply_kw_series, initial_capacity_mw, growth_rate
    except Exception as e: print(f"ERROR [_genSingleRenew]: Failed generating hourly series for profile '{profile_id}': {e}"); raise

def load_renewable_supply_from_city_data(
    city_solar_capacity_points_dict, city_wind_capacity_points_dict,
    solar_profile_id, wind_profile_id,
    renewable_profiles_df, config):
    print(f"INFO [RenewSupply]: Loading COMBINED supply. Solar Prof: '{solar_profile_id}', Wind Prof: '{wind_profile_id}'")
    if renewable_profiles_df is None or renewable_profiles_df.empty:
        print("ERROR [RenewSupply]: Renewable profiles DataFrame is missing or empty.")
        return pd.DataFrame(), 0.0, 0.0
    try:
        solar_kw_series, initial_solar_mw, solar_growth = _generate_single_renewable_series(
            city_solar_capacity_points_dict, solar_profile_id, renewable_profiles_df, config)
        wind_kw_series, initial_wind_mw, wind_growth = _generate_single_renewable_series(
            city_wind_capacity_points_dict, wind_profile_id, renewable_profiles_df, config)

        if not solar_kw_series.index.equals(wind_kw_series.index):
             print("WARN [RenewSupply]: Solar and Wind series indices mismatch! Attempting reindex.")
             common_index = solar_kw_series.index.union(wind_kw_series.index)
             solar_kw_series = solar_kw_series.reindex(common_index, fill_value=0.0)
             wind_kw_series = wind_kw_series.reindex(common_index, fill_value=0.0)

        total_supply_kw = solar_kw_series + wind_kw_series
        supply_df = pd.DataFrame({'supply_kw': total_supply_kw})
        total_initial_mw = initial_solar_mw + initial_wind_mw
        forecast_years = config['FORECAST_YEARS']
        final_solar_mw_est = initial_solar_mw * ((1.0 + solar_growth) ** forecast_years) if initial_solar_mw > 1e-6 else 0.0
        final_wind_mw_est = initial_wind_mw * ((1.0 + wind_growth) ** forecast_years) if initial_wind_mw > 1e-6 else 0.0
        final_total_mw_est = max(0.0, final_solar_mw_est) + max(0.0, final_wind_mw_est)
        print(f"  Combined Supply Generated. Initial MW: {total_initial_mw:.2f} (S:{initial_solar_mw:.2f}, W:{initial_wind_mw:.2f}). Est Final MW: {final_total_mw_est:.2f}")
        return supply_df, total_initial_mw, final_total_mw_est
    except Exception as e:
        print(f"ERROR [RenewSupply]: Failed generating combined hourly supply series: {e}"); traceback.print_exc(); return pd.DataFrame(), 0.0, 0.0

def simulate_energy_balance_and_charging(soh_forecast_df, fleet_size_initial, supply_df, config):
    start_balance = timer.time()
    fleet_growth_rate = config['FLEET_GROWTH_RATE']; daily_km_per_ev = config['AVG_DAILY_KM_PER_EV']; base_efficiency_kwh_km = config['BASE_EFFICIENCY_KWH_KM']; charger_mix = config['CHARGER_MIX']; resistance_impact_factor = config['RESISTANCE_IMPACT_FACTOR']; charge_start_hour = config['CHARGE_START_HOUR']; charge_end_hour = config['CHARGE_END_HOUR']; initial_max_grid_power_kw = config['INITIAL_MAX_GRID_POWER_KW']; grid_limit_growth_rate = config['GRID_LIMIT_GROWTH_RATE']; use_grid_storage = config['USE_GRID_STORAGE']; storage_capacity_kwh = config['STORAGE_CAPACITY_KWH']; storage_max_charge_kw = config['STORAGE_MAX_CHARGE_KW']; storage_max_discharge_kw = config['STORAGE_MAX_DISCHARGE_KW']; storage_efficiency = config['STORAGE_EFFICIENCY']; forecast_years = config['FORECAST_YEARS']; initial_soh_res = config['INITIAL_AVG_RESISTANCE_PERCENT']; initial_soh_cap = config['INITIAL_AVG_SOH_PERCENT']

    if supply_df is None or supply_df.empty or 'supply_kw' not in supply_df.columns:
        print("ERROR [EnergyBalance]: Invalid supply DataFrame provided.")
        return None
    try:
        hourly_index = pd.to_datetime(supply_df.index)
        supply_df.index = hourly_index 
    except Exception as e:
        print(f"ERROR [EnergyBalance]: supply_df index is not convertible to datetime: {e}")
        return None
    if len(hourly_index) < 1:
        print("ERROR [EnergyBalance]: Hourly index derived from supply_df has zero length.")
        return None

    start_date = hourly_index.min(); sim_start_year = start_date.year

    chg_hrs = list(range(charge_start_hour, charge_end_hour)) if charge_start_hour < charge_end_hour else list(range(charge_start_hour, 24)) + list(range(0, charge_end_hour))
    chg_hrs = list(range(24)) if charge_start_hour == charge_end_hour else chg_hrs
    num_charge_hours_per_day = len(chg_hrs)
    assert num_charge_hours_per_day > 0, "Charge window cannot be zero hours! Check CHARGE_START_HOUR and CHARGE_END_HOUR."

    try:
        daily_idx = pd.date_range(start=start_date.normalize(), end=hourly_index.max().normalize(), freq='D')
        daily_state = pd.DataFrame(index=daily_idx)
    except Exception as e:
        print(f"ERROR [EnergyBalance]: Failed to create daily index: {e}")
        return None

    soh_valid = False
    if soh_forecast_df is not None and not soh_forecast_df.empty and all(c in soh_forecast_df for c in ["Year","Avg_SoH_Cap_Percent","Resistance_Percent"]):
        try:
            soh_idx = soh_forecast_df.set_index('Year')
            if soh_idx[["Avg_SoH_Cap_Percent", "Resistance_Percent"]].isnull().any().any():
                 print("WARN [EnergyBalance]: NaNs found in SoH forecast data.")
                 soh_valid = True 
            else:
                 soh_valid = True
        except Exception as e:
            print(f"WARN [EnergyBalance]: Error processing SoH forecast input: {e}")
            soh_valid = False 
    if not soh_valid:
        print("WARN [EnergyBalance]: Using initial SoH/Res values throughout due to invalid/missing forecast input.")
        soh_data = [{"Year": yr, "Avg_SoH_Cap_Percent": initial_soh_cap, "Resistance_Percent": initial_soh_res} for yr in range(forecast_years + 1)]
        soh_idx = pd.DataFrame(soh_data).set_index('Year')

    daily_state['year_offset'] = (daily_state.index.year - sim_start_year)
    try:
        daily_state['avg_soh_cap'] = soh_idx['Avg_SoH_Cap_Percent'].reindex(daily_state['year_offset']).ffill().values
        daily_state['resistance_pct'] = soh_idx['Resistance_Percent'].reindex(daily_state['year_offset']).ffill().values
    except Exception as e:
        print(f"ERROR [EnergyBalance]: Failed mapping SoH data to daily state: {e}")
        return None

    try:
        fleet_size_yearly = {yr_off: int(max(0, fleet_size_initial * ((1.0 + fleet_growth_rate) ** yr_off))) for yr_off in range(forecast_years + 1)}
        daily_state['fleet_size'] = daily_state['year_offset'].map(fleet_size_yearly).ffill().astype(int)
        daily_state.ffill(inplace=True) 
        if daily_state.isnull().any().any():
            print("ERROR [EnergyBalance]: NaNs found in daily_state DataFrame before simulation loop.")
            print(daily_state.isnull().sum())
            return None
        daily_kwh_per_ev = daily_km_per_ev * base_efficiency_kwh_km
        daily_state['total_daily_fleet_kwh_need'] = daily_state['fleet_size'] * daily_kwh_per_ev
    except Exception as e:
        print(f"ERROR [EnergyBalance]: Failed calculating daily fleet size or energy need: {e}")
        return None

    stor_soc = 0.0 
    total_fraction = sum(m.get('fraction', 0) for m in charger_mix.values())
    if abs(total_fraction - 1.0) > 1e-3:
        if total_fraction < 1e-6: print("ERROR [EnergyBalance]: Charger mix fractions sum to zero or less."); return None
        print(f"WARN [EnergyBalance]: Charger mix fractions sum to {total_fraction:.3f}, normalizing.")
        for ch_key in charger_mix: charger_mix[ch_key]['fraction'] /= total_fraction 
    avg_chg_kw = sum(m['fraction'] * m['power'] for m in charger_mix.values())
    charge_eff = max(0.0, min(1.0, storage_efficiency)); charge_eff_inv = 1.0 / charge_eff if charge_eff > 1e-9 else float('inf')
    results_list = []; yearly_max_grid_kw = { sim_start_year + yr: initial_max_grid_power_kw * ((1.0 + grid_limit_growth_rate) ** yr) for yr in range(forecast_years + 1) }
    supply_kw_arr = supply_df['supply_kw'].values; hourly_dt_index = supply_df.index; hourly_hour_arr = hourly_dt_index.hour.values;
    daily_state.index = pd.to_datetime(daily_state.index)
    daily_state_dict = daily_state.to_dict('index') 
    cols = ['fleet_size','avg_soh_cap','resistance_pct','supply_kw','max_grid_power_kw','unused_renewables_kw','total_hourly_potential_charge_need_kw','energy_charged_this_hour_kwh','demand_met_by_renewables_kwh','demand_met_by_storage_kwh','demand_met_by_grid_kwh','charge_to_storage_from_renewables_kwh','charge_to_storage_from_grid_kwh','unmet_demand_kwh','grid_draw_kw','storage_soc_kwh']
    target_charge_power_kw = 0.0; available_renewables_kw = 0.0; rem_ev_need = 0.0; current_timestamp = hourly_index[0] 

    print(f"DEBUG [EnergyBalance]: Starting loop for {len(hourly_index)} hours.") 
    loop_entered = False

    for i in range(len(hourly_index)):
        try:
            loop_entered = True
            current_timestamp = hourly_dt_index[i]; hr = hourly_hour_arr[i]
            current_day_key = current_timestamp.normalize()

            daily_info = daily_state_dict.get(current_day_key)

            if daily_info is None:
                if i % 5000 == 0: 
                    print(f"DEBUG [EnergyBalance]: daily_info is None for index {i}, hourly_ts {current_timestamp}, lookup_key {current_day_key}. Skipping hour.")
                    if i == 0: print(f"  Example keys in daily_state_dict: {list(daily_state_dict.keys())[:5]}") 
                continue 

            if i == 0 or (i + 1) % 8760 == 0: 
                print(f"DEBUG [EnergyBalance]: Processing hour {i+1}/{len(hourly_index)}, Timestamp: {current_timestamp}")
                print(f"  ---> Retrieved Daily Info: Keys={list(daily_info.keys()) if daily_info else 'None'}")

            f_s = daily_info['fleet_size']; soh_c = daily_info['avg_soh_cap']; res_p = daily_info['resistance_pct']; daily_total_need_kwh = daily_info['total_daily_fleet_kwh_need']; current_max_grid_kw = yearly_max_grid_kw.get(current_timestamp.year, initial_max_grid_power_kw);
            if i < len(supply_kw_arr): available_renewables_kw = supply_kw_arr[i]
            else: print(f"WARN [EnergyBalance]: Index {i} out of bounds for supply_kw_arr. Setting supply to 0."); available_renewables_kw = 0.0

            hourly_result = {key: 0.0 for key in cols}; hourly_result.update({'fleet_size': f_s, 'avg_soh_cap': soh_c, 'resistance_pct': res_p, 'max_grid_power_kw': current_max_grid_kw, 'supply_kw': available_renewables_kw })

            target_charge_power_kw = 0.0
            if hr in chg_hrs and daily_total_need_kwh > 1e-6 and f_s > 0:
                ideal_power_needed_per_ev = (daily_kwh_per_ev / num_charge_hours_per_day)
                res_increase_factor = max(0.0, (max(initial_soh_res, res_p) - initial_soh_res) / initial_soh_res if initial_soh_res > 1e-6 else 0.0)
                res_penalty_div = 1.0 + (res_increase_factor * resistance_impact_factor)
                effective_avg_chg_power_per_ev = (avg_chg_kw / res_penalty_div) if res_penalty_div > 1e-6 else 0.0
                max_fleet_power_draw_kw = f_s * effective_avg_chg_power_per_ev; ideal_fleet_power_need_kw = f_s * ideal_power_needed_per_ev
                target_charge_power_kw = max(0.0, min(ideal_fleet_power_need_kw, max_fleet_power_draw_kw))
            hourly_result['total_hourly_potential_charge_need_kw'] = target_charge_power_kw

            ev_ren = ev_sto = ev_grid = 0.0; sto_ren_gross_kw = sto_ren_net_kwh = 0.0; sto_grid_gross_kw = sto_grid_net_kwh = 0.0; grid_ev_kw = grid_sto_kw = 0.0
            rem_ev_need = target_charge_power_kw; rem_ren = available_renewables_kw; rem_grid_limit_kw = current_max_grid_kw
            ev_ren = min(rem_ev_need, rem_ren); rem_ev_need -= ev_ren; rem_ren -= ev_ren
            if use_grid_storage and rem_ren > 0 and stor_soc < storage_capacity_kwh:
                space_kwh = storage_capacity_kwh - stor_soc; chg_lim_rate = storage_max_charge_kw; chg_lim_space_kw = max(0.0, space_kwh * charge_eff_inv) if charge_eff > 1e-9 else 0.0
                sto_ren_gross_kw = min(rem_ren, chg_lim_rate, chg_lim_space_kw); sto_ren_net_kwh = (sto_ren_gross_kw * charge_eff) if sto_ren_gross_kw > 1e-9 else 0.0
                stor_soc += sto_ren_net_kwh; rem_ren -= sto_ren_gross_kw; stor_soc = min(stor_soc, storage_capacity_kwh)
            hourly_result['unused_renewables_kw'] = max(0.0, rem_ren)
            if use_grid_storage and rem_ev_need > 0 and stor_soc > 1e-9:
                dis_lim_rate = storage_max_discharge_kw; dis_lim_soc_kwh = max(0.0, stor_soc); ev_sto_kw = min(rem_ev_need, dis_lim_rate, dis_lim_soc_kwh)
                ev_sto = ev_sto_kw; stor_soc -= ev_sto; rem_ev_need -= ev_sto; stor_soc = max(0.0, stor_soc)
            if rem_ev_need > 0 and rem_grid_limit_kw > 0:
                grid_ev_kw = min(rem_ev_need, rem_grid_limit_kw); ev_grid = grid_ev_kw; rem_ev_need -= ev_grid; rem_grid_limit_kw -= grid_ev_kw
            if use_grid_storage and hr in chg_hrs and rem_grid_limit_kw > 0 and stor_soc < storage_capacity_kwh:
                space_kwh = storage_capacity_kwh - stor_soc; chg_lim_rate = storage_max_charge_kw; chg_lim_space_kw = max(0.0, space_kwh * charge_eff_inv) if charge_eff > 1e-9 else 0.0
                sto_grid_gross_kw = min(rem_grid_limit_kw, chg_lim_rate, chg_lim_space_kw); sto_grid_net_kwh = (sto_grid_gross_kw * charge_eff) if sto_grid_gross_kw > 1e-9 else 0.0
                grid_sto_kw = sto_grid_gross_kw; stor_soc += sto_grid_net_kwh; stor_soc = min(stor_soc, storage_capacity_kwh)

            total_charged_kwh = ev_ren + ev_sto + ev_grid; total_grid_draw_kw = grid_ev_kw + grid_sto_kw; unmet_kwh = max(0.0, rem_ev_need)
            hourly_result.update({'energy_charged_this_hour_kwh': total_charged_kwh, 'demand_met_by_renewables_kwh': ev_ren, 'demand_met_by_storage_kwh': ev_sto, 'demand_met_by_grid_kwh': ev_grid, 'charge_to_storage_from_renewables_kwh': sto_ren_net_kwh if use_grid_storage else 0.0, 'charge_to_storage_from_grid_kwh': sto_grid_net_kwh if use_grid_storage else 0.0, 'grid_draw_kw': total_grid_draw_kw, 'unmet_demand_kwh': unmet_kwh, 'storage_soc_kwh': stor_soc if use_grid_storage else 0.0})

            results_list.append(hourly_result)

        except Exception as loop_error: 
            print(f"\n!!!!!! ERROR inside energy balance loop at index {i}, timestamp {current_timestamp} !!!!!!")
            print(f"Error Type: {type(loop_error).__name__}"); print(f"Error Details: {loop_error}")
            print(f"  Current Storage SoC (kWh): {stor_soc}")
            print(f"  Target Charge Power (kW): {target_charge_power_kw}")
            print(f"  Available Renewables (kW): {available_renewables_kw}")
            print(f"  Remaining EV Need (kWh): {rem_ev_need}")
            if daily_info: print(f"  Daily Info: Fleet={daily_info.get('fleet_size')}, Need={daily_info.get('total_daily_fleet_kwh_need')}, SoH={daily_info.get('avg_soh_cap'):.2f}, Res={daily_info.get('resistance_pct'):.2f}")
            else: print("  Daily Info: Was None")
            traceback.print_exc(); return None 

    if not loop_entered: 
        print("ERROR [EnergyBalance]: Loop body was never entered. Check hourly_index length.")
    if not results_list: 
        print("ERROR [EnergyBalance]: results_list is empty after loop completed without errors.")
        return pd.DataFrame() 

    try:
        results_df = pd.DataFrame(results_list, index=hourly_index)
        results_df.fillna(0.0, inplace=True)
        numeric_cols = results_df.select_dtypes(include=np.number).columns
        results_df[numeric_cols] = results_df[numeric_cols].clip(lower=0) 
        results_df['year'] = results_df.index.year
    except Exception as df_error:
        print(f"ERROR [EnergyBalance]: Failed creating final DataFrame from results_list: {df_error}")
        return None 

    print(f"  Hourly energy balance loop completed in {timer.time() - start_balance:.2f} sec.")
    return results_df
def calculate_financial_metrics(results_df, initial_solar_mw, initial_wind_mw, config, city_factors):
    print("INFO [Financial]: Calculating Annual Financials & LCOE (Combined Solar/Wind)...")
    annual_financials_list = []; overall_summary = {}
    try:
        base_capex_solar = config['CAPEX_SOLAR_USD_PER_KW'] * 1000; base_capex_wind = config['CAPEX_WIND_USD_PER_KW'] * 1000; base_capex_storage_kwh = config['CAPEX_STORAGE_USD_PER_KWH']; base_capex_storage_kw = config['CAPEX_STORAGE_USD_PER_KW']; base_opex_solar_pct = config['OPEX_SOLAR_PERCENT_CAPEX'] / 100.0; base_opex_wind_pct = config['OPEX_WIND_PERCENT_CAPEX'] / 100.0; base_opex_storage_pct = config['OPEX_STORAGE_PERCENT_CAPEX'] / 100.0
        discount_rate = config['DISCOUNT_RATE']; lifetime = config['ANALYSIS_LIFETIME_YEARS']; use_storage = config['USE_GRID_STORAGE']; storage_cap_kwh = config['STORAGE_CAPACITY_KWH']; storage_power_kw = config['STORAGE_MAX_CHARGE_KW']
        grid_price = city_factors.get('grid_price_usd_kwh', config['GRID_PRICE_USD_PER_KWH']); capex_mult = city_factors.get('capex_mult', 1.0); opex_mult = city_factors.get('opex_mult', 1.0)
        print(f"  Using City Factors: Grid Price={grid_price:.3f}, CAPEX Mult={capex_mult:.2f}, OPEX Mult={opex_mult:.2f}")

        adj_capex_solar_mw = base_capex_solar * capex_mult; adj_capex_wind_mw = base_capex_wind * capex_mult; adj_capex_storage_kwh = base_capex_storage_kwh * capex_mult; adj_capex_storage_kw = base_capex_storage_kw * capex_mult; adj_opex_solar_pct = base_opex_solar_pct * opex_mult; adj_opex_wind_pct = base_opex_wind_pct * opex_mult; adj_opex_storage_pct = base_opex_storage_pct * opex_mult

        initial_solar_capex = max(0.0, initial_solar_mw) * adj_capex_solar_mw
        initial_wind_capex = max(0.0, initial_wind_mw) * adj_capex_wind_mw
        initial_ren_capex = initial_solar_capex + initial_wind_capex
        initial_storage_capex = 0
        if use_storage and storage_cap_kwh > 0 and storage_power_kw > 0:
             storage_capex_kwh_part = storage_cap_kwh * adj_capex_storage_kwh; storage_capex_kw_part = storage_power_kw * adj_capex_storage_kw; initial_storage_capex = storage_capex_kwh_part + storage_capex_kw_part
        total_initial_capex = initial_ren_capex + initial_storage_capex

        sim_start_year = results_df['year'].min(); analysis_end_year = sim_start_year + lifetime - 1
        sim_years_available = sorted(results_df['year'].unique()); analysis_years_in_sim = [y for y in sim_years_available if (y - sim_start_year) < lifetime]
        if not analysis_years_in_sim: print("WARN [Financial]: No simulation years fall within the analysis lifetime. Financials will be limited.")

        total_discounted_cost_pv = total_initial_capex; total_energy_charged_kwh_lifetime = 0; cumulative_cost = total_initial_capex; cumulative_savings = 0
        annual_financials_list.append({"Year": sim_start_year - 1, "Analysis_Year": 0, "Annual_Solar_OPEX": 0, "Annual_Wind_OPEX": 0, "Annual_Ren_OPEX": 0, "Annual_Sto_OPEX": 0, "Annual_Grid_Cost": 0, "Total_Annual_Cost": 0, "Discounted_Annual_Cost": 0, "Annual_Energy_Charged_kWh": 0, "Baseline_Annual_Grid_Cost": 0, "Annual_Savings": 0, "Cumulative_Cost": cumulative_cost, "Cumulative_Savings": cumulative_savings, "Cumulative_Net_Position": cumulative_savings - cumulative_cost})

        for year_idx, year in enumerate(analysis_years_in_sim, 1):
            year_df = results_df[results_df['year'] == year];
            if year_df.empty: continue
            annual_solar_opex = initial_solar_capex * adj_opex_solar_pct; annual_wind_opex = initial_wind_capex * adj_opex_wind_pct; annual_ren_opex = annual_solar_opex + annual_wind_opex; annual_sto_opex = initial_storage_capex * adj_opex_storage_pct if use_storage else 0; annual_opex = annual_ren_opex + annual_sto_opex
            annual_grid_kwh = year_df['demand_met_by_grid_kwh'].sum(); annual_grid_cost = annual_grid_kwh * grid_price
            current_year_cost = annual_opex + annual_grid_cost; discount_factor = (1 + discount_rate) ** year_idx; discounted_annual_cost = current_year_cost / discount_factor; total_discounted_cost_pv += discounted_annual_cost
            annual_energy_charged = year_df['energy_charged_this_hour_kwh'].sum(); total_energy_charged_kwh_lifetime += annual_energy_charged
            baseline_annual_grid_cost = annual_energy_charged * grid_price; annual_savings = baseline_annual_grid_cost - current_year_cost
            cumulative_cost += current_year_cost; cumulative_savings += annual_savings
            annual_financials_list.append({"Year": year, "Analysis_Year": year_idx, "Annual_Solar_OPEX": annual_solar_opex, "Annual_Wind_OPEX": annual_wind_opex, "Annual_Ren_OPEX": annual_ren_opex, "Annual_Sto_OPEX": annual_sto_opex, "Annual_Grid_Cost": annual_grid_cost, "Total_Annual_Cost": current_year_cost, "Discounted_Annual_Cost": discounted_annual_cost, "Annual_Energy_Charged_kWh": annual_energy_charged, "Baseline_Annual_Grid_Cost": baseline_annual_grid_cost, "Annual_Savings": annual_savings, "Cumulative_Cost": cumulative_cost, "Cumulative_Savings": cumulative_savings, "Cumulative_Net_Position": cumulative_savings - cumulative_cost})

        lcoe = total_discounted_cost_pv / total_energy_charged_kwh_lifetime if total_energy_charged_kwh_lifetime > 1e-6 else 0
        REASONABLE_LCOE_THRESHOLD = 0.50;
        if lcoe > REASONABLE_LCOE_THRESHOLD: print(f"WARN [Financial]: LCOE ({lcoe:.3f} USD/kWh) > {REASONABLE_LCOE_THRESHOLD}.")
        elif lcoe < 0: print(f"WARN [Financial]: Calculated LCOE is negative ({lcoe:.3f} USD/kWh).")
        total_savings_lifetime = annual_financials_list[-1]['Cumulative_Savings'] if len(annual_financials_list) > 1 else 0
        total_cost_lifetime_undiscounted = annual_financials_list[-1]['Cumulative_Cost'] if len(annual_financials_list) > 1 else total_initial_capex

        overall_summary = {
            "LCOE_USD_per_kWh_Lifetime": lcoe, "Baseline_Grid_Only_Cost_USD_Lifetime": total_energy_charged_kwh_lifetime * grid_price,
            "Total_System_Discounted_Cost_USD_Lifetime": total_discounted_cost_pv, "Total_System_Undiscounted_Cost_USD_Lifetime": total_cost_lifetime_undiscounted,
            "Total_Undiscounted_Savings_USD_Lifetime": total_savings_lifetime, "Financial_Lifetime_Assumed_Years": lifetime, "Financial_Discount_Rate": discount_rate,
            "Financial_Grid_Price_Applied_USD_per_kWh": grid_price, "Financial_CAPEX_Multiplier_Applied": capex_mult, "Financial_OPEX_Multiplier_Applied": opex_mult,
            "Financial_Initial_Solar_MW": initial_solar_mw, "Financial_Initial_Wind_MW": initial_wind_mw, "Financial_Initial_Total_Renew_MW": initial_solar_mw + initial_wind_mw,
            "Financial_Initial_Solar_CAPEX_USD": initial_solar_capex, "Financial_Initial_Wind_CAPEX_USD": initial_wind_capex,
            "Financial_Initial_Storage_CAPEX_USD": initial_storage_capex, "Financial_Initial_Total_CAPEX_USD": total_initial_capex,
            "Financial_Renewable_Type_Assumed": "Mixed Solar/Wind",
        }
        print(f"  Financial Metrics Calculated: LCOE={lcoe:.4f}, Total Undiscounted Savings={total_savings_lifetime:,.0f}")
        annual_financials_df = pd.DataFrame(annual_financials_list)
        return overall_summary, annual_financials_df
    except Exception as e: print(f"ERROR [Financial]: Calculation failed: {e}"); traceback.print_exc(); return {}, pd.DataFrame()

def plot_cumulative_financials(annual_fin_df, city_name="", config=None):
    if annual_fin_df is None or annual_fin_df.empty or 'Cumulative_Net_Position' not in annual_fin_df.columns: return None
    try:
        fig, ax = _setup_plot((10, 6), f'Cumulative Net Financial Position ({city_name})', 'Year of Operation', 'Cumulative Net Savings (USD)')
        df_plot = annual_fin_df[annual_fin_df['Analysis_Year'] >= 0].copy()
        if 'Analysis_Year' not in df_plot or 'Cumulative_Net_Position' not in df_plot: print(f"WARN [PlotCumFin]: Missing required columns for {city_name}."); plt.close(fig); return None
        years = df_plot['Analysis_Year']; net_position = df_plot['Cumulative_Net_Position']
        ax.plot(years, net_position, marker='o', linestyle='-', color='green', label='Cum. Net Savings'); ax.axhline(0, color='grey', linestyle='--', lw=1, label='Breakeven'); payback_year = np.nan
        if not net_position.empty and (net_position >= 0).any(): 
            try:
                payback_idx_arr = np.where(net_position.values >= 0)[0]
                if len(payback_idx_arr) > 0:
                    payback_idx = payback_idx_arr[0]
                    if payback_idx == 0: payback_year = years.iloc[payback_idx] 
                    else: 
                        y1 = net_position.iloc[payback_idx-1]; y2 = net_position.iloc[payback_idx]; x1 = years.iloc[payback_idx-1]; x2 = years.iloc[payback_idx]
                        if abs(y2 - y1) > 1e-9: payback_year = x1 + (x2 - x1) * (0 - y1) / (y2 - y1)
                        else: payback_year = x2 
                    if not np.isnan(payback_year): ax.axvline(payback_year, color='red', linestyle=':', lw=1.5, label=f'Est. Payback: ~Yr {payback_year:.1f}')
            except Exception as e_inner: print(f"WARN [PlotCumFin]: Error during payback calc for {city_name}: {e_inner}")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f')); ax.tick_params(axis='y', labelrotation=45); ax.grid(True, linestyle=':', alpha=0.7); ax.legend(loc='best'); fig.tight_layout(); return fig
    except Exception as e: print(f"ERROR [PlotCumFin]: Failed plotting cumulative financials for {city_name}: {e}"); plt.close(fig) if 'fig' in locals() else None; return None

def plot_annual_costs(annual_fin_df, city_name="", config=None):
    if annual_fin_df is None or annual_fin_df.empty: return None
    cost_cols = ['Annual_Solar_OPEX', 'Annual_Wind_OPEX', 'Annual_Sto_OPEX', 'Annual_Grid_Cost']
    available_cols = [c for c in cost_cols if c in annual_fin_df.columns and annual_fin_df[c].sum() > 1e-9]
    if not available_cols or 'Analysis_Year' not in annual_fin_df.columns: print(f"WARN [PlotAnnCosts]: Missing required cost columns or 'Analysis_Year' for {city_name}."); return None
    try:
        fig, ax = _setup_plot((12, 6), f'Annual System Costs ({city_name})', 'Year of Operation', 'Annual Cost (USD)')
        df_plot = annual_fin_df[annual_fin_df['Analysis_Year'] >= 1].copy()
        if df_plot.empty: print(f"WARN [PlotAnnCosts]: No data available for analysis years >= 1 for {city_name}."); plt.close(fig); return None
        years = df_plot['Analysis_Year']; pos = np.arange(len(years)); plot_data = df_plot[available_cols]
        colors = {'Annual_Solar_OPEX': '#ffcc66', 'Annual_Wind_OPEX': '#99ccff', 'Annual_Sto_OPEX': '#aec6cf', 'Annual_Grid_Cost': '#ff6961'}
        labels = {'Annual_Solar_OPEX': 'Solar OPEX', 'Annual_Wind_OPEX': 'Wind OPEX', 'Annual_Sto_OPEX': 'Storage OPEX', 'Annual_Grid_Cost': 'Grid Energy Cost'}
        bottom = np.zeros(len(years))
        for col in available_cols:
             values_to_plot = plot_data[col].fillna(0).values; ax.bar(pos, values_to_plot, bottom=bottom, label=labels.get(col, col), color=colors.get(col, '#cccccc'), edgecolor='grey', width=0.8); bottom += values_to_plot
        initial_capex = 0
        if 'summary' in config and isinstance(config['summary'], dict): 
             initial_capex = config['summary'].get('Financial_Initial_Total_CAPEX_USD', 0)
        elif 'Analysis_Year' in annual_fin_df.columns and 0 in annual_fin_df['Analysis_Year'].values:
            capex_row = annual_fin_df.loc[annual_fin_df['Analysis_Year'] == 0]
            if not capex_row.empty and 'Cumulative_Cost' in capex_row.columns:
                 try: initial_capex = float(capex_row['Cumulative_Cost'].iloc[0])
                 except (ValueError, TypeError): initial_capex = 0
        ax.set_title(f'Annual System Costs ({city_name})\nInitial CAPEX (Year 0): ${initial_capex:,.0f}', fontsize=12)
        ax.set_xticks(pos); ax.set_xticklabels(years.astype(int)); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f')); ax.tick_params(axis='y', labelrotation=45); ax.legend(title="Cost Component"); ax.grid(True, axis='y', linestyle=':'); ax.set_ylim(bottom=0); fig.tight_layout(); return fig
    except Exception as e: print(f"ERROR [PlotAnnCosts]: Failed plotting annual costs for {city_name}: {e}"); plt.close(fig) if 'fig' in locals() else None; return None

def plot_annual_savings(annual_fin_df, city_name="", config=None):
    if annual_fin_df is None or annual_fin_df.empty or 'Annual_Savings' not in annual_fin_df.columns: return None
    try:
        fig, ax = _setup_plot((10, 5), f'Annual Undiscounted Savings vs. Grid-Only ({city_name})', 'Year of Operation', 'Annual Savings (USD)')
        df_plot = annual_fin_df[annual_fin_df['Analysis_Year'] >= 1].copy(); years = df_plot['Analysis_Year']; savings = df_plot['Annual_Savings']; pos = np.arange(len(years)); colors = ['green' if s >= 0 else 'red' for s in savings]; ax.bar(pos, savings, color=colors, edgecolor='grey', width=0.8); ax.axhline(0, color='grey', linestyle='--', lw=1); ax.set_xticks(pos); ax.set_xticklabels(years.astype(int)); ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f')); ax.tick_params(axis='y', labelrotation=45); ax.grid(True, axis='y', linestyle=':'); fig.tight_layout(); return fig
    except Exception as e: print(f"ERROR [PlotAnnSav]: {e}"); return None

def plot_city_comparison(summary1, summary2, fin_df1, fin_df2, city1, city2, config):
    print(f"INFO [PlotComp]: Generating comparison plots for {city1} vs {city2}...")
    figs = [] 
    def add_comparison_bars(ax, x_locs, vals1, vals2, width, label1, label2, color1='skyblue', color2='lightcoral'):
        vals1_num = [v if isinstance(v, (int, float)) and not np.isnan(v) else 0 for v in vals1]; vals2_num = [v if isinstance(v, (int, float)) and not np.isnan(v) else 0 for v in vals2]
        rects1 = ax.bar(x_locs - width/2, vals1_num, width, label=label1, color=color1, edgecolor='black'); rects2 = ax.bar(x_locs + width/2, vals2_num, width, label=label2, color=color2, edgecolor='black'); return rects1, rects2
    def add_bar_value_labels(ax, rects1, rects2):
         try: ax.bar_label(rects1, padding=3, fmt='%.3g', fontsize=8, label_type='edge', color='black', weight='normal', labels=[f'{v:.3g}' if not np.isnan(v) and abs(v)>1e-9 else '0' for v in rects1.datavalues])
         except Exception as e: print(f"WARN: Could not add labels to rects1: {e}")
         try: ax.bar_label(rects2, padding=3, fmt='%.3g', fontsize=8, label_type='edge', color='black', weight='normal', labels=[f'{v:.3g}' if not np.isnan(v) and abs(v)>1e-9 else '0' for v in rects2.datavalues])
         except Exception as e: print(f"WARN: Could not add labels to rects2: {e}")

    metrics_fig1 = {'Energy Charged (GWh)': 'Total_Energy_Charged_GWh', 'Peak Grid Load (MW)': 'Peak_Grid_Load_MW'}
    labels1 = []; city1_vals1, city2_vals1 = [], []; v1=v2=None
    for label, key in metrics_fig1.items(): v1, v2 = summary1.get(key), summary2.get(key);
    if isinstance(v1,(int,float)) and isinstance(v2,(int,float)) and not (np.isnan(v1) or np.isnan(v2)): labels1.append(label); city1_vals1.append(v1); city2_vals1.append(v2)
    if labels1:
        try: fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5)); x1 = np.arange(len(labels1)); width = 0.35; rects1_1, rects1_2 = add_comparison_bars(ax1, x1, city1_vals1, city2_vals1, width, city1, city2); ax1.set_ylabel('Value (GWh or MW)'); ax1.set_title(f'Energy & Power Comparison: {city1} vs {city2}'); ax1.set_xticks(x1); ax1.set_xticklabels(labels1, rotation=15, ha='right'); ax1.legend(); ax1.grid(True, axis='y', linestyle=':'); add_bar_value_labels(ax1, rects1_1, rects1_2); fig1.tight_layout(); figs.append(fig1)
        except Exception as e: print(f"ERROR [PlotComp Fig1]: {e}")

    metrics_fig2 = {'Unmet Demand (%)': 'Unmet_Need_Percent', 'Renewable Utilization (%)': 'Renewable_Utilization_%'}
    labels2 = []; city1_vals2, city2_vals2 = [], []; v1=v2=None
    for label, key in metrics_fig2.items(): v1, v2 = summary1.get(key), summary2.get(key);
    if isinstance(v1,(int,float)) and isinstance(v2,(int,float)) and not (np.isnan(v1) or np.isnan(v2)): labels2.append(label); city1_vals2.append(v1); city2_vals2.append(v2)
    if labels2:
        try: fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5)); x2 = np.arange(len(labels2)); width = 0.35; rects2_1, rects2_2 = add_comparison_bars(ax2, x2, city1_vals2, city2_vals2, width, city1, city2); ax2.set_ylabel('Value (%)'); ax2.set_title(f'Performance Comparison: {city1} vs {city2}'); ax2.set_xticks(x2); ax2.set_xticklabels(labels2, rotation=15, ha='right'); ax2.legend(); ax2.grid(True, axis='y', linestyle=':'); ax2.set_ylim(bottom=min(0, min(city1_vals2)*1.1 if city1_vals2 else 0, min(city2_vals2)*1.1 if city2_vals2 else 0 )); add_bar_value_labels(ax2, rects2_1, rects2_2); fig2.tight_layout(); figs.append(fig2)
        except Exception as e: print(f"ERROR [PlotComp Fig2]: {e}")

    metrics_fleet = {'Final Fleet Size': 'Final_Fleet_Size_Est'};
    metrics_cap = {'Initial Total Renew Cap (MW)': 'Initial_Total_Renewable_Capacity_MW_Derived', 'Final Total Renew Cap (MW)': 'Final_Total_Renewable_Capacity_MW_Est'}
    has_fleet = any(k in summary1 and k in summary2 for k in metrics_fleet.values()); has_cap = any(k in summary1 and k in summary2 for k in metrics_cap.values())
    if has_fleet or has_cap:
         try:
             num_subplots = (1 if has_fleet else 0) + (1 if has_cap else 0); assert num_subplots > 0, "No fleet/cap data"
             fig3, axes3 = plt.subplots(num_subplots, 1, figsize=(8, num_subplots * 4.5), squeeze=False); axes3 = axes3.flatten(); subplot_idx = 0; fig3.suptitle(f'Physical Size Comparison: {city1} vs {city2}', y=1.0 if num_subplots > 1 else 1.02)
             if has_fleet:
                 ax_fleet = axes3[subplot_idx]; labels_fleet = []; city1_fleet, city2_fleet = [], []; v1=v2=None
                 for label, key in metrics_fleet.items(): v1, v2 = summary1.get(key), summary2.get(key);
                 if isinstance(v1,(int,float)) and isinstance(v2,(int,float)) and not (np.isnan(v1) or np.isnan(v2)): labels_fleet.append(label); city1_fleet.append(v1); city2_fleet.append(v2)
                 if labels_fleet: x_fleet = np.arange(len(labels_fleet)); width = 0.35; r3a_1, r3a_2 = add_comparison_bars(ax_fleet, x_fleet, city1_fleet, city2_fleet, width, city1, city2); ax_fleet.set_xticks(x_fleet); ax_fleet.set_xticklabels(labels_fleet); ax_fleet.set_ylabel('Vehicle Count'); ax_fleet.set_title('Fleet Size'); ax_fleet.grid(True, axis='y', linestyle=':'); ax_fleet.legend(); add_bar_value_labels(ax_fleet, r3a_1, r3a_2)
                 subplot_idx += 1
             if has_cap:
                 ax_cap = axes3[subplot_idx]; labels_cap = []; city1_cap, city2_cap = [], []; v1=v2=None
                 for label, key in metrics_cap.items(): v1, v2 = summary1.get(key), summary2.get(key);
                 if isinstance(v1,(int,float)) and isinstance(v2,(int,float)) and not (np.isnan(v1) or np.isnan(v2)): labels_cap.append(label); city1_cap.append(v1); city2_cap.append(v2)
                 if labels_cap: x_cap = np.arange(len(labels_cap)); width = 0.35; r3b_1, r3b_2 = add_comparison_bars(ax_cap, x_cap, city1_cap, city2_cap, width, city1, city2); ax_cap.set_xticks(x_cap); ax_cap.set_xticklabels(labels_cap, rotation=15, ha='right'); ax_cap.set_ylabel('Capacity (MW)'); ax_cap.set_title('Total Renewable Capacity'); ax_cap.grid(True, axis='y', linestyle=':'); ax_cap.legend(); add_bar_value_labels(ax_cap, r3b_1, r3b_2)
             fig3.tight_layout(rect=[0, 0.03, 1, 0.95]); figs.append(fig3)
         except Exception as e: print(f"ERROR [PlotComp Fig3 - Physical Size]: {e}")

    metrics_fig4 = {'LCOE ($/kWh)': 'LCOE_USD_per_kWh_Lifetime', 'Total Lifetime Savings ($)': 'Total_Undiscounted_Savings_USD_Lifetime'}
    metrics_lcoe = {k:v for k,v in metrics_fig4.items() if 'LCOE' in k}; metrics_savings = {k:v for k,v in metrics_fig4.items() if 'Savings' in k}
    has_finance = any(k in summary1 and k in summary2 for k in metrics_fig4.values())
    if has_finance:
        try:
            fig4, axes4 = plt.subplots(2, 1, figsize=(8, 8)); fig4.suptitle(f'Overall Financial Comparison: {city1} vs {city2}\n(Based on CONFIG assumptions)', y=1.0)
            ax4_lcoe = axes4[0]; labels_lcoe = []; city1_lcoe, city2_lcoe = [], []; v1=v2=None
            for label, key in metrics_lcoe.items(): v1, v2 = summary1.get(key), summary2.get(key);
            if isinstance(v1,(int,float)) and isinstance(v2,(int,float)) and not (np.isnan(v1) or np.isnan(v2)): labels_lcoe.append(label); city1_lcoe.append(v1); city2_lcoe.append(v2)
            if labels_lcoe: x_lcoe = np.arange(len(labels_lcoe)); width = 0.35; r4a_1, r4a_2 = add_comparison_bars(ax4_lcoe, x_lcoe, city1_lcoe, city2_lcoe, width, city1, city2); ax4_lcoe.set_xticks(x_lcoe); ax4_lcoe.set_xticklabels(labels_lcoe); ax4_lcoe.set_ylabel('Value ($/kWh)'); ax4_lcoe.set_title('Levelized Cost of Energy (LCOE)'); ax4_lcoe.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.3f')); ax4_lcoe.grid(True, axis='y', linestyle=':'); ax4_lcoe.legend(); add_bar_value_labels(ax4_lcoe, r4a_1, r4a_2)
            ax4_sav = axes4[1]; labels_sav = []; city1_sav, city2_sav = [], []; v1=v2=None
            for label, key in metrics_savings.items(): v1, v2 = summary1.get(key), summary2.get(key);
            if isinstance(v1,(int,float)) and isinstance(v2,(int,float)) and not (np.isnan(v1) or np.isnan(v2)): labels_sav.append(label); city1_sav.append(v1); city2_sav.append(v2)
            if labels_sav: x_sav = np.arange(len(labels_sav)); width = 0.35; r4b_1, r4b_2 = add_comparison_bars(ax4_sav, x_sav, city1_sav, city2_sav, width, city1, city2); ax4_sav.set_xticks(x_sav); ax4_sav.set_xticklabels(labels_sav); ax4_sav.set_ylabel('Value ($)'); ax4_sav.set_title('Total Undiscounted Lifetime Savings vs Grid'); ax4_sav.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f')); ax4_sav.tick_params(axis='y', labelrotation=30); ax4_sav.grid(True, axis='y', linestyle=':'); ax4_sav.legend(); add_bar_value_labels(ax4_sav, r4b_1, r4b_2)
            fig4.tight_layout(rect=[0, 0.03, 1, 0.95]); figs.append(fig4)
        except Exception as e: print(f"ERROR [PlotComp Fig4 - Financial]: {e}")

    try:
        if fin_df1 is not None and not fin_df1.empty and fin_df2 is not None and not fin_df2.empty and 'Cumulative_Net_Position' in fin_df1 and 'Annual_Savings' in fin_df1 and 'Cumulative_Net_Position' in fin_df2 and 'Annual_Savings' in fin_df2:
            fig5, axes5 = plt.subplots(2, 1, figsize=(10, 9), sharex=True); fig5.suptitle(f'Financial Trends Comparison: {city1} vs {city2}', y=1.0)
            ax5_cum = axes5[0]; df1_plot_cum = fin_df1[fin_df1['Analysis_Year'] >= 0]; df2_plot_cum = fin_df2[fin_df2['Analysis_Year'] >= 0]; max_year = max(df1_plot_cum['Analysis_Year'].max(), df2_plot_cum['Analysis_Year'].max()) if not df1_plot_cum.empty and not df2_plot_cum.empty else 0; years_aligned = np.arange(0, max_year + 1); net1 = np.interp(years_aligned, df1_plot_cum['Analysis_Year'], df1_plot_cum['Cumulative_Net_Position'], left=np.nan, right=np.nan); net2 = np.interp(years_aligned, df2_plot_cum['Analysis_Year'], df2_plot_cum['Cumulative_Net_Position'], left=np.nan, right=np.nan); ax5_cum.plot(years_aligned, net1, marker='.', linestyle='-', color='skyblue', label=f'{city1} Cum. Net Savings'); ax5_cum.plot(years_aligned, net2, marker='.', linestyle='--', color='lightcoral', label=f'{city2} Cum. Net Savings'); ax5_cum.axhline(0, color='grey', linestyle='--', lw=1, label='Breakeven'); ax5_cum.set_ylabel('Cumulative Savings ($)'); ax5_cum.set_title('Cumulative Net Financial Position (Payback)'); ax5_cum.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f')); ax5_cum.tick_params(axis='y', labelrotation=30); ax5_cum.grid(True, linestyle=':', alpha=0.7); ax5_cum.legend(loc='best', fontsize='small')
            ax5_ann = axes5[1]; df1_plot_ann = fin_df1[fin_df1['Analysis_Year'] >= 1]; df2_plot_ann = fin_df2[fin_df2['Analysis_Year'] >= 1]; max_year_ann = max(df1_plot_ann['Analysis_Year'].max(), df2_plot_ann['Analysis_Year'].max()) if not df1_plot_ann.empty and not df2_plot_ann.empty else 0; years_aligned_ann = np.arange(1, max_year_ann + 1); savings1 = df1_plot_ann.set_index('Analysis_Year')['Annual_Savings'].reindex(years_aligned_ann).fillna(0).values; savings2 = df2_plot_ann.set_index('Analysis_Year')['Annual_Savings'].reindex(years_aligned_ann).fillna(0).values; x_ann = np.arange(len(years_aligned_ann)); width_ann = 0.35; rects5a = ax5_ann.bar(x_ann - width_ann/2, savings1, width_ann, label=f'{city1} Annual Savings', color='skyblue', edgecolor='darkgrey'); rects5b = ax5_ann.bar(x_ann + width_ann/2, savings2, width_ann, label=f'{city2} Annual Savings', color='lightcoral', edgecolor='darkgrey'); ax5_ann.axhline(0, color='grey', linestyle='--', lw=1); ax5_ann.set_xlabel('Year of Operation'); ax5_ann.set_ylabel('Annual Savings ($)'); ax5_ann.set_title('Annual Undiscounted Savings vs. Grid-Only'); ax5_ann.set_xticks(x_ann); ax5_ann.set_xticklabels(years_aligned_ann.astype(int)); ax5_ann.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f')); ax5_ann.tick_params(axis='y', labelrotation=30); ax5_ann.grid(True, axis='y', linestyle=':'); ax5_ann.legend(loc='best', fontsize='small');
            fig5.tight_layout(rect=[0, 0.03, 1, 0.95]); figs.append(fig5)
    except Exception as e: print(f"WARN [PlotComp Fig5 - Financial Trends]: {e}"); traceback.print_exc()

    print(f"INFO [PlotComp]: Generated {len(figs)} comparison figures.")
    return figs

@functools.lru_cache(maxsize=32)
def run_full_city_simulation(
    city_name, avg_temp_c, population, city_solar_capacity_tuple,
    city_wind_capacity_tuple, solar_profile_id, wind_profile_id,
    global_config_tuple ):
    sim_start_time = timer.time()
    print(f"\n------ CACHE MISS/RUN: Running COMBINED S/W Simulation for {city_name} ------")
    global renewable_profiles_data_df_global
    assert renewable_profiles_data_df_global is not None and not renewable_profiles_data_df_global.empty, "Global renewable profiles DF not loaded."

    try: 
        keys = [ "FORECAST_YEARS", "FLEET_GROWTH_RATE", "AVG_DAILY_KM_PER_EV", "BASE_EFFICIENCY_KWH_KM", "CYCLES_PER_DAY", "CHEMISTRY", "TRAFFIC_LEVEL", "USER_PROFILE", "PYBAMM_PARAM_SET", "PYBAMM_SIM_TYPE", "PYBAMM_SIM_CRATE", "HEAT_TRANSFER_COEFF", "PYBAMM_INITIAL_SOC", "CHARGER_MIX_JSON", "RESISTANCE_IMPACT_FACTOR", "CHARGE_START_HOUR", "CHARGE_END_HOUR", "SMART_CHARGING_STRATEGY", "USE_GRID_STORAGE", "STORAGE_CAPACITY_KWH", "STORAGE_MAX_CHARGE_KW", "STORAGE_MAX_DISCHARGE_KW", "STORAGE_EFFICIENCY", "INITIAL_AVG_SOH_PERCENT", "INITIAL_AVG_RESISTANCE_PERCENT", "BATTERY_NOMINAL_CAPACITY_KWH", "PROFILE_ID_COL", "PROFILE_HOUR_COL", "PROFILE_VALUE_COL", "INITIAL_MAX_GRID_POWER_KW", "GRID_LIMIT_GROWTH_RATE", "RENEW_CAPACITY_YEARS_TUPLE", "SIM_START_YEAR", "CAPEX_SOLAR_USD_PER_KW", "CAPEX_WIND_USD_PER_KW", "CAPEX_STORAGE_USD_PER_KWH", "CAPEX_STORAGE_USD_PER_KW", "OPEX_SOLAR_PERCENT_CAPEX", "OPEX_WIND_PERCENT_CAPEX", "OPEX_STORAGE_PERCENT_CAPEX", "GRID_PRICE_USD_PER_KWH", "DISCOUNT_RATE", "ANALYSIS_LIFETIME_YEARS", "MIN_INITIAL_FLEET_SIZE", "EV_ADOPTION_RATE_PER_CAPITA"]
        assert len(global_config_tuple) == len(keys), f"Config tuple length mismatch. Expected {len(keys)}, got {len(global_config_tuple)}\nKEYS:{keys}\nTUPLE:{global_config_tuple}"
        local_config = dict(zip(keys, global_config_tuple))
        local_config['CHARGER_MIX'] = json.loads(local_config['CHARGER_MIX_JSON'])
        local_config['RENEW_CAPACITY_YEARS'] = list(local_config['RENEW_CAPACITY_YEARS_TUPLE'])
    except Exception as e: raise ValueError(f"Error unpacking/reconstructing config tuple: {e}") from e

    city_solar_capacity_points = dict(city_solar_capacity_tuple)
    city_wind_capacity_points = dict(city_wind_capacity_tuple)

    try:
        initial_fleet_size = max(local_config.get('MIN_INITIAL_FLEET_SIZE', 10), int(population * local_config.get('EV_ADOPTION_RATE_PER_CAPITA', 0.01)))
        rep_avg_temp, rep_peak_temp = run_pybamm_simulation(ambient_temp_c=avg_temp_c)
        assert not (pd.isna(rep_avg_temp) or pd.isna(rep_peak_temp)), f"PyBaMM lookup failed for {city_name}"
        soh_history_df, soh_err = forecast_fleet_degradation(years=local_config['FORECAST_YEARS'], initial_soh_cap=local_config['INITIAL_AVG_SOH_PERCENT'], initial_soh_res_percent=local_config['INITIAL_AVG_RESISTANCE_PERCENT'], cycles_per_day=local_config['CYCLES_PER_DAY'], avg_temp_c=rep_avg_temp, peak_temp_c=rep_peak_temp, traffic_level=local_config['TRAFFIC_LEVEL'], user_profile=local_config['USER_PROFILE'], chemistry=local_config['CHEMISTRY'], config=local_config)
        if soh_err: print(f"WARN [Sim]: Degradation forecast warning: {soh_err}")
        assert not soh_history_df.empty and not soh_history_df[['Avg_SoH_Cap_Percent', 'Resistance_Percent']].isnull().any().any(), "Invalid degradation forecast results."
        final_soh_cap = soh_history_df['Avg_SoH_Cap_Percent'].iloc[-1]; final_res_pct = soh_history_df['Resistance_Percent'].iloc[-1]

        renewable_supply_df, city_init_total_mw, city_final_total_mw = load_renewable_supply_from_city_data(city_solar_capacity_points_dict=city_solar_capacity_points, city_wind_capacity_points_dict=city_wind_capacity_points, solar_profile_id=solar_profile_id, wind_profile_id=wind_profile_id, renewable_profiles_df=renewable_profiles_data_df_global, config=local_config)
        assert renewable_supply_df is not None and not renewable_supply_df.empty, "Combined renewable supply generation failed."

        sim_start_year = local_config['SIM_START_YEAR']
        years_solar = sorted(city_solar_capacity_points.keys()); mws_solar = [max(0.0, float(city_solar_capacity_points.get(y, 0.0))) for y in years_solar]
        initial_solar_mw_derived = float(np.interp(sim_start_year, years_solar, mws_solar, left=mws_solar[0], right=mws_solar[-1])) if years_solar else 0.0
        years_wind = sorted(city_wind_capacity_points.keys()); mws_wind = [max(0.0, float(city_wind_capacity_points.get(y, 0.0))) for y in years_wind]
        initial_wind_mw_derived = float(np.interp(sim_start_year, years_wind, mws_wind, left=mws_wind[0], right=mws_wind[-1])) if years_wind else 0.0

        results_df = simulate_energy_balance_and_charging(soh_forecast_df=soh_history_df, fleet_size_initial=initial_fleet_size, supply_df=renewable_supply_df, config=local_config)
        assert results_df is not None and not results_df.empty, "Energy balance simulation failed."

        GWH = 1e6; KWH_PER_EV_BATT = local_config.get('BATTERY_NOMINAL_CAPACITY_KWH', 60); charged = results_df['energy_charged_this_hour_kwh'].sum(); unmet = results_df['unmet_demand_kwh'].sum(); need = max(charged + unmet, 1e-9); ren_direct = results_df['demand_met_by_renewables_kwh'].sum(); stor_dis = results_df['demand_met_by_storage_kwh'].sum(); grid = results_df['demand_met_by_grid_kwh'].sum(); total_renewable_supply_kwh = results_df['supply_kw'].sum(); curtailed = results_df['unused_renewables_kw'].sum(); ren_to_sto = results_df.get('charge_to_storage_from_renewables_kwh', pd.Series(0.0)).sum(); sim_hours = len(results_df); sim_days = sim_hours / 24.0 if sim_hours > 0 else 1; avg_daily_sim_supply_kwh = total_renewable_supply_kwh / sim_days; avg_daily_sim_need_kwh = need / sim_days; sim_supply_equiv_evs = avg_daily_sim_supply_kwh / KWH_PER_EV_BATT if KWH_PER_EV_BATT > 1e-9 else 0.0; sim_need_equiv_evs = avg_daily_sim_need_kwh / KWH_PER_EV_BATT if KWH_PER_EV_BATT > 1e-9 else 0.0; final_fleet_est = initial_fleet_size * ((1.0 + local_config['FLEET_GROWTH_RATE'])**local_config['FORECAST_YEARS']); final_total_ren_cap_est = city_final_total_mw

        summary = {
            "City": city_name, "Avg_Ambient_Temp_C": avg_temp_c, "Population": population, "Sim_Avg_Batt_Temp_C": rep_avg_temp, "Sim_Peak_Batt_Temp_C": rep_peak_temp,
            "Initial_Fleet_Size_Heuristic": initial_fleet_size, "Initial_Solar_Capacity_MW_Derived": initial_solar_mw_derived, "Initial_Wind_Capacity_MW_Derived": initial_wind_mw_derived,
            "Initial_Total_Renewable_Capacity_MW_Derived": city_init_total_mw, "Final_Total_Renewable_Capacity_MW_Est": final_total_ren_cap_est,
            "Final_Fleet_Size_Est": final_fleet_est, "Final_Fleet_Avg_SoH_Cap_%": final_soh_cap, "Final_Fleet_Avg_Resistance_%": final_res_pct,
            "Total_Simulated_EV_Need_GWh": need / GWH, "Total_Energy_Charged_GWh": charged / GWH, "Total_Unmet_Demand_GWh": unmet / GWH, "Unmet_Need_Percent": (unmet / need * 100),
            f"Avg_Daily_Total_EV_Need_kWh_Sim": avg_daily_sim_need_kwh, f"Avg_Daily_Sim_Supply_kWh": avg_daily_sim_supply_kwh, f"Equivalent_EVs_per_Day_Based_on_Sim_Need (assuming {KWH_PER_EV_BATT}kWh/EV)": sim_need_equiv_evs, f"Equivalent_EVs_per_Day_Based_on_Sim_Supply (assuming {KWH_PER_EV_BATT}kWh/EV)": sim_supply_equiv_evs,
            "Peak_Grid_Load_MW": results_df['grid_draw_kw'].max() / 1000.0 if not results_df.empty else 0.0,
            "Charged_from_Direct_Renewables_GWh": ren_direct / GWH, "Charged_from_Storage_Discharge_GWh": stor_dis / GWH, "Charged_from_Grid_GWh": grid / GWH,
            "Charged_from_Direct_Renewables_%": (ren_direct / charged * 100) if charged > 1e-9 else 0.0, "Charged_from_Storage_Discharge_%": (stor_dis / charged * 100) if charged > 1e-9 else 0.0, "Charged_from_Grid_%": (grid / charged * 100) if charged > 1e-9 else 0.0,
            "Total_Renewable_Supply_GWh": total_renewable_supply_kwh / GWH, "Total_Renewable_Used_GWh": (ren_direct + ren_to_sto) / GWH, "Total_Renewable_Curtailed_GWh": curtailed / GWH,
            "Renewable_Utilization_%": ((ren_direct + ren_to_sto) / total_renewable_supply_kwh * 100) if total_renewable_supply_kwh > 1e-9 else 0.0,
            "Renewables_Direct_to_EV_GWh": ren_direct / GWH, "Renewables_to_Storage_GWh": ren_to_sto / GWH,
            "Solar_Profile_ID_Used": solar_profile_id, "Wind_Profile_ID_Used": wind_profile_id
        }

        city_factors = get_city_financial_factors(city_name, local_config)
        financial_summary, annual_financials_df = calculate_financial_metrics(results_df=results_df, initial_solar_mw=initial_solar_mw_derived, initial_wind_mw=initial_wind_mw_derived, config=local_config, city_factors=city_factors)
        summary.update(financial_summary)

        summary = {k: (v if not (isinstance(v, (float, int, np.number)) and np.isnan(v)) else 0.0) for k, v in summary.items()}
        elapsed = timer.time() - sim_start_time
        print(f"------ Full Simulation for {city_name} completed successfully in {elapsed:.2f} seconds ------")
        return summary, soh_history_df, results_df, annual_financials_df
    except Exception as sim_error:
        print(f"\n!!!!!! ERROR during cached simulation run for '{city_name}': {type(sim_error).__name__}: {sim_error} !!!!!!")
        traceback.print_exc()
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def _load_and_clean_data(config):
    global cities_df_global, city_renewable_capacity_df_global, renewable_profiles_data_df_global
    all_data_valid = True

    try:
        path = config['CITIES_CSV_PATH']; print(f"\nLoading city list: {path}")
        df_cities = pd.read_csv(path); cols = [config['CITY_COL'], config['TEMP_COL'], config['POP_COL']]; assert all(c in df_cities.columns for c in cols), f"Missing city cols: {cols}"
        df_cities[config['TEMP_COL']] = pd.to_numeric(df_cities[config['TEMP_COL']], errors='coerce'); df_cities[config['POP_COL']] = pd.to_numeric(df_cities[config['POP_COL']], errors='coerce')
        df_cities.dropna(subset=cols, inplace=True); df_cities = df_cities[df_cities[config['POP_COL']] > 0].reset_index(drop=True); df_cities[config['POP_COL']] = df_cities[config['POP_COL']].astype(int)
        df_cities[config['CITY_COL']] = df_cities[config['CITY_COL']].astype(str).str.strip().str.title(); df_cities = df_cities.drop_duplicates(subset=[config['CITY_COL']], keep='first')
        assert not df_cities.empty, "No valid cities found after cleaning."; cities_df_global = df_cities; print(f"Loaded {len(df_cities)} valid cities.")
    except Exception as e: print(f"ERROR loading cities list from '{config.get('CITIES_CSV_PATH')}': {e}"); all_data_valid = False

    try:
        path = config['CITY_RENEWABLE_CAPACITY_PATH']; print(f"Loading city capacity (Solar/Wind): {path}")
        df_cap = pd.read_csv(path)
        required_cap_cols = [config['RENEW_CITY_COL'], config['RENEW_SOLAR_PROFILE_ID_COL'], config['RENEW_WIND_PROFILE_ID_COL']] + config['ALL_RENEW_MW_COLS']
        missing_cols = [c for c in required_cap_cols if c not in df_cap.columns]; assert not missing_cols, f"Missing required capacity columns: {missing_cols}"
        df_cap.dropna(subset=[config['RENEW_CITY_COL'], config['RENEW_SOLAR_PROFILE_ID_COL'], config['RENEW_WIND_PROFILE_ID_COL']], inplace=True)
        df_cap[config['RENEW_CITY_COL']] = df_cap[config['RENEW_CITY_COL']].astype(str).str.strip().str.title()
        df_cap[config['RENEW_SOLAR_PROFILE_ID_COL']] = df_cap[config['RENEW_SOLAR_PROFILE_ID_COL']].astype(str).str.strip()
        df_cap[config['RENEW_WIND_PROFILE_ID_COL']] = df_cap[config['RENEW_WIND_PROFILE_ID_COL']].astype(str).str.strip()
        for mw_col in config['ALL_RENEW_MW_COLS']:
            df_cap[mw_col] = pd.to_numeric(df_cap[mw_col], errors='coerce')
            if (df_cap[mw_col] < 0).any(): print(f"WARN: Found negative values in '{mw_col}'. Clipping to 0."); df_cap[mw_col] = df_cap[mw_col].clip(lower=0.0)
        df_cap = df_cap.drop_duplicates(subset=[config['RENEW_CITY_COL']], keep='first')
        assert not df_cap.empty, "No valid capacity data found after cleaning."; city_renewable_capacity_df_global = df_cap; print(f"Loaded capacity data for {len(df_cap)} unique cities (Solar/Wind).")
    except Exception as e: print(f"ERROR loading city capacity from '{config.get('CITY_RENEWABLE_CAPACITY_PATH')}': {e}"); traceback.print_exc(); all_data_valid = False

    try:
        path = config['RENEWABLE_PROFILES_DATA_PATH']; print(f"Loading renewable profiles: {path}")
        df_prof = pd.read_csv(path); cols = [config['PROFILE_ID_COL'], config['PROFILE_HOUR_COL'], config['PROFILE_VALUE_COL']]; assert all(c in df_prof.columns for c in cols), f"Missing profile cols: {cols}"
        df_prof.dropna(subset=[config['PROFILE_ID_COL'], config['PROFILE_HOUR_COL']], inplace=True)
        df_prof[config['PROFILE_ID_COL']] = df_prof[config['PROFILE_ID_COL']].astype(str).str.strip()
        df_prof[config['PROFILE_HOUR_COL']] = pd.to_numeric(df_prof[config['PROFILE_HOUR_COL']], errors='coerce')
        df_prof[config['PROFILE_VALUE_COL']] = pd.to_numeric(df_prof[config['PROFILE_VALUE_COL']], errors='coerce')
        initial_rows = len(df_prof); df_prof.dropna(subset=[config['PROFILE_HOUR_COL'], config['PROFILE_VALUE_COL']], inplace=True)
        if len(df_prof) < initial_rows: print(f"WARN: Dropped {initial_rows - len(df_prof)} rows from profiles (non-numeric hour/value or missing ID/hour initially).")
        assert not df_prof.empty, "No valid profile data remaining after cleaning."
        renewable_profiles_data_df_global = df_prof; print(f"Loaded {len(df_prof)} valid profile rows for {df_prof[config['PROFILE_ID_COL']].nunique()} profiles.")
    except Exception as e: print(f"ERROR loading renewable profiles from '{config.get('RENEWABLE_PROFILES_DATA_PATH')}': {e}"); all_data_valid = False

    return all_data_valid

def _create_global_config_tuple(config_dict):
    try:
        required_keys = [ "FORECAST_YEARS", "FLEET_GROWTH_RATE", "AVG_DAILY_KM_PER_EV", "BASE_EFFICIENCY_KWH_KM", "CYCLES_PER_DAY", "CHEMISTRY", "TRAFFIC_LEVEL", "USER_PROFILE", "PYBAMM_PARAM_SET", "PYBAMM_SIM_TYPE", "PYBAMM_SIM_CRATE", "HEAT_TRANSFER_COEFF", "PYBAMM_INITIAL_SOC", "CHARGER_MIX", "RESISTANCE_IMPACT_FACTOR", "CHARGE_START_HOUR", "CHARGE_END_HOUR", "SMART_CHARGING_STRATEGY", "USE_GRID_STORAGE", "STORAGE_CAPACITY_KWH", "STORAGE_MAX_CHARGE_KW", "STORAGE_MAX_DISCHARGE_KW", "STORAGE_EFFICIENCY", "INITIAL_AVG_SOH_PERCENT", "INITIAL_AVG_RESISTANCE_PERCENT", "BATTERY_NOMINAL_CAPACITY_KWH", "PROFILE_ID_COL", "PROFILE_HOUR_COL", "PROFILE_VALUE_COL", "INITIAL_MAX_GRID_POWER_KW", "GRID_LIMIT_GROWTH_RATE", "RENEW_CAPACITY_YEARS", "SIM_START_YEAR", "CAPEX_SOLAR_USD_PER_KW", "CAPEX_WIND_USD_PER_KW", "CAPEX_STORAGE_USD_PER_KWH", "CAPEX_STORAGE_USD_PER_KW", "OPEX_SOLAR_PERCENT_CAPEX", "OPEX_WIND_PERCENT_CAPEX", "OPEX_STORAGE_PERCENT_CAPEX", "GRID_PRICE_USD_PER_KWH", "DISCOUNT_RATE", "ANALYSIS_LIFETIME_YEARS", "MIN_INITIAL_FLEET_SIZE", "EV_ADOPTION_RATE_PER_CAPITA" ]
        missing_keys = [k for k in required_keys if k not in config_dict]; assert not missing_keys, f"Missing keys required for config tuple: {missing_keys}"
        config_tuple_list = []
        for key in required_keys:
            value = config_dict[key]
            if key == "CHARGER_MIX": config_tuple_list.append(json.dumps(value, sort_keys=True))
            elif key == "RENEW_CAPACITY_YEARS": config_tuple_list.append(tuple(sorted(value)))
            else: config_tuple_list.append(value)
        return tuple(config_tuple_list)
    except Exception as e: raise ValueError(f"Failed to create immutable config tuple: {e}") from e

def _acquire_lock(lock_path, retries=5, wait=0.5):
    for attempt in range(retries):
        try: lock_fd = open(lock_path, 'x'); lock_fd.close(); return True
        except FileExistsError:
            if attempt < retries - 1: timer.sleep(wait)
            else: print(f"WARN [Save]: Lock file '{lock_path.name}' exists. Skipping save.")
        except Exception as e: print(f"ERROR [Save]: Error acquiring lock: {e}")
    return False

def _release_lock(lock_path):
    if lock_path.exists():
        try: lock_path.unlink()
        except OSError as e: print(f"WARN [Save]: Could not remove lock file '{lock_path}': {e}")

def _save_aggregate_summary(summary_dict, city_name, config):
    summary_path = Path(config['SUMMARY_OUTPUT_FILENAME'])
    lock_path = summary_path.with_suffix('.lock')
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(summary_dict, dict) or not summary_dict: print(f"WARN [Save]: Invalid or empty summary for {city_name}. Skipping save."); return
    if 'City' not in summary_dict: summary_dict['City'] = city_name

    if _acquire_lock(lock_path):
        try:
            summary_df_all = pd.DataFrame(); header = True
            if summary_path.is_file() and summary_path.stat().st_size > 0:
                try: summary_df_all = pd.read_csv(summary_path); header = False if not summary_df_all.empty else True
                except pd.errors.EmptyDataError: print(f"WARN [Save]: Summary file '{summary_path}' empty. Overwriting."); header = True
                except Exception as e: print(f"WARN [Save]: Error reading existing summary '{summary_path}': {e}. Overwriting."); header = True
            current_summary_df = pd.DataFrame([summary_dict])
            if not summary_df_all.empty and 'City' in summary_df_all.columns:
                all_cols = summary_df_all.columns.union(current_summary_df.columns); summary_df_all = summary_df_all.reindex(columns=all_cols); current_summary_df = current_summary_df.reindex(columns=all_cols)
                summary_df_all = summary_df_all[summary_df_all['City'] != city_name].copy(); summary_df_all = pd.concat([summary_df_all, current_summary_df], ignore_index=True, sort=False)
            else: summary_df_all = current_summary_df
            for col in summary_df_all.columns: 
                 if summary_df_all[col].apply(lambda x: isinstance(x, (list, tuple, dict))).any(): summary_df_all[col] = summary_df_all[col].astype(str)
            summary_df_all.to_csv(summary_path, index=False, float_format='%.5g', header=header) 
            print(f"\nINFO [Save]: Aggregate summary updated: {summary_path.resolve()}")
        except Exception as e_save: print(f"\nERROR [Save]: Could not process/save aggregate summary: {e_save}"); traceback.print_exc()
        finally: _release_lock(lock_path)

def _save_hourly_results(results_df, city_name, config):
    if results_df is None or results_df.empty: return
    safe_city_name = "".join(c if c.isalnum() else "_" for c in city_name).rstrip('_').replace("__","_"); output_dir = config['HOURLY_RESULTS_DIR']; output_dir.mkdir(parents=True, exist_ok=True)
    cols_to_save = [ 'fleet_size', 'avg_soh_cap', 'resistance_pct', 'supply_kw', 'max_grid_power_kw', 'unused_renewables_kw', 'total_hourly_potential_charge_need_kw', 'energy_charged_this_hour_kwh', 'demand_met_by_renewables_kwh', 'demand_met_by_storage_kwh', 'demand_met_by_grid_kwh', 'charge_to_storage_from_renewables_kwh', 'charge_to_storage_from_grid_kwh', 'unmet_demand_kwh', 'grid_draw_kw', 'storage_soc_kwh', 'year' ]
    cols_exist = [c for c in cols_to_save if c in results_df.columns]; df_to_save = results_df[cols_exist].copy();
    if isinstance(df_to_save.index, pd.DatetimeIndex) and df_to_save.index.tz is not None: df_to_save.index = df_to_save.index.tz_localize(None)
    save_successful = False; file_version = "v3.6" 
    if PYARROW_AVAILABLE:
        try: f_path = output_dir / f"{safe_city_name}_hourly_results_{file_version}.parquet"; df_to_save.to_parquet(f_path, index=True, engine='pyarrow'); print(f"INFO [Save]: Hourly saved (Parquet): {f_path.resolve()}"); save_successful = True
        except Exception as e: print(f"WARN [Save]: Parquet save failed for {city_name}: {e}. Trying CSV."); save_successful = False
    if not save_successful:
        try: f_path = output_dir / f"{safe_city_name}_hourly_results_{file_version}.csv"; df_to_save.to_csv(f_path, index=True, float_format='%.5g'); reason = "PyArrow unavailable" if not PYARROW_AVAILABLE else "Parquet fallback"; print(f"INFO [Save]: Hourly saved (CSV - {reason}): {f_path.resolve()}")
        except Exception as e: print(f"ERROR [Save]: CSV save also failed for {city_name}: {e}")

def select_city(available_cities, prompt_num=""):
    target_city_name = ""
    print(f"\n--- City Selection {prompt_num}---")
    print(f"Available ({len(available_cities)}):")
    limit = 20
    sorted_available_cities = sorted(available_cities)
    print('\n'.join(f" - {c}" for c in sorted_available_cities[:limit]))
    if len(sorted_available_cities) > limit:
        print("   ...")

    while not target_city_name:
        try:
            user_input_raw = input(f"Enter city name {prompt_num}(or 'list' to see all): ").strip()
            if not user_input_raw: continue
            user_input = user_input_raw.title()
            if user_input.lower() == 'list':
                print("\nFull list of simulatable cities:"); print('\n'.join(f" - {c}" for c in sorted_available_cities)); continue
            if user_input in available_cities: target_city_name = user_input
            else:
                print(f"ERROR: City '{user_input_raw}' not found or not simulatable.")
                if FUZZY_AVAILABLE:
                    matches = [s[0] for s in process.extract(user_input, sorted_available_cities, limit=5, scorer=fuzz.token_sort_ratio) if s[1] > 70]
                    if matches: print("Did you mean:", ", ".join(matches), "?")
        except EOFError: print("\nInput stream closed. Exiting."); sys.exit(1)
    return target_city_name
def _setup_plot(figsize, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    return fig, ax

def plot_soh_forecast(soh_df, city_name="", config=None):
    if soh_df is None or soh_df.empty or not all(c in soh_df for c in ['Year', 'Avg_SoH_Cap_Percent', 'Resistance_Percent']):
        print(f"WARN [PlotSOH]: Invalid or incomplete SoH data for {city_name}. Skipping plot.")
        return None
    try:
        print(f"DEBUG [PlotSOH]: Plotting SoH for {city_name}...") 
        fig, ax1 = _setup_plot((10, 5), f'SoH Forecast ({city_name})', 'Simulation Year', 'SoH Capacity (%)')
        ax2 = ax1.twinx() 

        color1 = 'tab:blue'
        ax1.set_ylabel('SoH Capacity (%)', color=color1, fontsize=12)
        ax1.plot(soh_df['Year'], soh_df['Avg_SoH_Cap_Percent'], color=color1, marker='o', linestyle='-', label='Capacity SoH')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, linestyle=':', alpha=0.7, axis='y')
        ax1.set_ylim(bottom=0) 

        color2 = 'tab:red'
        ax2.set_ylabel('Resistance (%)', color=color2, fontsize=12)  
        ax2.plot(soh_df['Year'], soh_df['Resistance_Percent'], color=color2, marker='x', linestyle='--', label='Resistance %')
        ax2.tick_params(axis='y', labelcolor=color2)
        initial_res = config.get('INITIAL_AVG_RESISTANCE_PERCENT', 100.0) if config else 100.0
        ax2.set_ylim(bottom=max(0, initial_res * 0.9)) 

        fig.tight_layout()
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')
        return fig
    except Exception as e:
        print(f"ERROR [PlotSOH]: Failed plotting SoH for {city_name}: {e}")
        traceback.print_exc()
        plt.close(fig) if 'fig' in locals() else None
        return None

def plot_energy_balance_sample(results_df, city_name="", config=None, sample_days=7):
    if results_df is None or results_df.empty:
        print(f"WARN [PlotBalanceSample]: Results data empty for {city_name}. Skipping plot.")
        return None
    try:
        print(f"DEBUG [PlotBalanceSample]: Plotting energy balance sample for {city_name}...") 
        sample_df = results_df.head(sample_days * 24).copy()
        if sample_df.empty:
             print(f"WARN [PlotBalanceSample]: Not enough data for {sample_days}-day sample for {city_name}.")
             return None

        fig, ax = _setup_plot((14, 7), f'Hourly Energy Balance Sample ({sample_days} days) - {city_name}', 'Timestamp', 'Power (kW) / Energy (kWh)')

        ax.plot(sample_df.index, sample_df['supply_kw'], label='Renewable Supply (kW)', color='green', alpha=0.8, linewidth=1.5)
        ax.plot(sample_df.index, sample_df['total_hourly_potential_charge_need_kw'], label='Potential EV Demand (kW)', color='black', linestyle=':', alpha=0.8, linewidth=1.5)
        ax.plot(sample_df.index, sample_df['grid_draw_kw'], label='Grid Draw (kW)', color='red', alpha=0.7, linewidth=1.5)

        ev_sources = ['demand_met_by_renewables_kwh', 'demand_met_by_storage_kwh', 'demand_met_by_grid_kwh']
        colors_fill = ['lightgreen', 'lightblue', 'salmon']
        labels_fill = ['EV Charge from Renewables (kWh)', 'EV Charge from Storage (kWh)', 'EV Charge from Grid (kWh)']
        ax.stackplot(sample_df.index,
                     sample_df['demand_met_by_renewables_kwh'].fillna(0),
                     sample_df['demand_met_by_storage_kwh'].fillna(0),
                     sample_df['demand_met_by_grid_kwh'].fillna(0),
                     labels=labels_fill, colors=colors_fill, alpha=0.6)

        ax.legend(loc='upper left', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(bottom=0)

        if config and config.get('USE_GRID_STORAGE', False) and 'storage_soc_kwh' in sample_df.columns:
             ax2 = ax.twinx()
             storage_max = config.get('STORAGE_CAPACITY_KWH', 1)
             ax2.plot(sample_df.index, sample_df['storage_soc_kwh'], label='Storage SoC (kWh)', color='purple', linestyle='--', alpha=0.7)
             ax2.set_ylabel('Storage State of Charge (kWh)', color='purple')
             ax2.tick_params(axis='y', labelcolor='purple')
             ax2.set_ylim(0, storage_max * 1.05) 
             ax2.legend(loc='upper right', fontsize='small') 

        plt.xticks(rotation=30, ha='right')
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR [PlotBalanceSample]: Failed plotting sample balance for {city_name}: {e}")
        traceback.print_exc()
        plt.close(fig) if 'fig' in locals() else None
        return None

def plot_yearly_peaks(results_df, city_name="", config=None):
    if results_df is None or results_df.empty or not all(c in results_df for c in ['year', 'grid_draw_kw']):
        print(f"WARN [PlotYearlyPeaks]: Invalid data for {city_name}. Skipping plot.")
        return None
    try:
        print(f"DEBUG [PlotYearlyPeaks]: Plotting yearly peaks for {city_name}...") 
        yearly_peaks = results_df.groupby('year')['grid_draw_kw'].max() / 1000 
        if yearly_peaks.empty: print(f"WARN [PlotYearlyPeaks]: No yearly peak data found after grouping for {city_name}."); return None
        fig, ax = _setup_plot((8, 5), f'Peak Grid Draw per Year ({city_name})', 'Year', 'Peak Grid Draw (MW)')
        ax.bar(yearly_peaks.index, yearly_peaks.values, color='red', edgecolor='black')
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f MW'))
        ax.grid(True, axis='y', linestyle=':')
        ax.set_xticks(yearly_peaks.index)
        ax.set_xticklabels(yearly_peaks.index.astype(int))
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR [PlotYearlyPeaks]: Failed plotting yearly peaks for {city_name}: {e}")
        traceback.print_exc()
        plt.close(fig) if 'fig' in locals() else None
        return None

def plot_yearly_energy_contribution(results_df, city_name="", config=None):
    required_cols = ['year', 'demand_met_by_renewables_kwh', 'demand_met_by_storage_kwh', 'demand_met_by_grid_kwh']
    if results_df is None or results_df.empty or not all(c in results_df for c in required_cols):
        print(f"WARN [PlotEnergyContrib]: Invalid data for {city_name}. Skipping plot.")
        return None
    try:
        print(f"DEBUG [PlotEnergyContrib]: Plotting energy contribution for {city_name}...") 
        yearly_totals = results_df.groupby('year')[required_cols[1:]].sum() / 1e6 
        if yearly_totals.empty: print(f"WARN [PlotEnergyContrib]: No yearly energy data found after grouping for {city_name}."); return None

        fig, ax = _setup_plot((10, 6), f'Yearly EV Energy Charged by Source ({city_name})', 'Year', 'Energy Charged (GWh)')
        yearly_totals.plot(kind='bar', stacked=True, ax=ax,
                           color={'demand_met_by_renewables_kwh': 'lightgreen', 'demand_met_by_storage_kwh': 'lightblue', 'demand_met_by_grid_kwh': 'salmon'},
                           edgecolor='grey')
        ax.legend(['From Renewables', 'From Storage', 'From Grid'], title="Source")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f GWh'))
        ax.tick_params(axis='x', rotation=0)
        ax.grid(True, axis='y', linestyle=':')
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR [PlotEnergyContrib]: Failed plotting energy contribution for {city_name}: {e}")
        traceback.print_exc()
        plt.close(fig) if 'fig' in locals() else None
        return None

def plot_yearly_renewable_utilization(results_df, city_name="", config=None):
    required_cols = ['year', 'supply_kw', 'demand_met_by_renewables_kwh', 'charge_to_storage_from_renewables_kwh']
    if results_df is None or results_df.empty or not all(c in results_df for c in required_cols):
        print(f"WARN [PlotRenewUtil]: Invalid data for {city_name}. Skipping plot.")
        return None
    try:
        print(f"DEBUG [PlotRenewUtil]: Plotting renewable utilization for {city_name}...") 
        yearly_sums = results_df.groupby('year')[required_cols[1:]].sum()
        yearly_sums['total_supply_kwh'] = yearly_sums['supply_kw'] 
        yearly_sums['total_used_kwh'] = yearly_sums['demand_met_by_renewables_kwh'] + yearly_sums['charge_to_storage_from_renewables_kwh']
        yearly_sums['utilization_pct'] = (yearly_sums['total_used_kwh'] / yearly_sums['total_supply_kwh'].replace(0, np.nan)) * 100
        yearly_sums['utilization_pct'].fillna(0, inplace=True)
        if yearly_sums.empty: print(f"WARN [PlotRenewUtil]: No yearly utilization data found after grouping for {city_name}."); return None

        fig, ax = _setup_plot((8, 5), f'Yearly Renewable Energy Utilization ({city_name})', 'Year', 'Utilization (%)')
        ax.bar(yearly_sums.index, yearly_sums['utilization_pct'], color='darkgreen', edgecolor='black')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax.grid(True, axis='y', linestyle=':')
        ax.set_ylim(0, 105)
        ax.set_xticks(yearly_sums.index)
        ax.set_xticklabels(yearly_sums.index.astype(int))
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR [PlotRenewUtil]: Failed plotting renewable utilization for {city_name}: {e}")
        traceback.print_exc()
        plt.close(fig) if 'fig' in locals() else None
        return None

def plot_average_daily_charging_profile(results_df, city_name="", config=None):
    required_cols = ['energy_charged_this_hour_kwh']
    if results_df is None or results_df.empty or not all(c in results_df for c in required_cols):
        print(f"WARN [PlotAvgDaily]: Invalid data for {city_name}. Skipping plot.")
        return None
    try:
        print(f"DEBUG [PlotAvgDaily]: Plotting average daily profile for {city_name}...") 
        results_df_copy = results_df.copy() 
        results_df_copy['hour'] = results_df_copy.index.hour
        avg_daily = results_df_copy.groupby('hour')['energy_charged_this_hour_kwh'].mean()
        if avg_daily.empty: print(f"WARN [PlotAvgDaily]: No average daily data found after grouping for {city_name}."); return None

        fig, ax = _setup_plot((10, 5), f'Average Daily EV Charging Profile ({city_name})', 'Hour of Day', 'Average Energy Charged (kWh)')
        avg_daily.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
        ax.grid(True, axis='y', linestyle=':')
        ax.set_xticks(range(24))
        ax.set_xticklabels(range(24))
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR [PlotAvgDaily]: Failed plotting average daily profile for {city_name}: {e}")
        traceback.print_exc()
        plt.close(fig) if 'fig' in locals() else None
        return None

def plot_yearly_unmet_demand(results_df, city_name="", config=None):
    required_cols = ['year', 'unmet_demand_kwh', 'energy_charged_this_hour_kwh']
    if results_df is None or results_df.empty or not all(c in results_df for c in required_cols):
        print(f"WARN [PlotUnmet]: Invalid data for {city_name}. Skipping plot.")
        return None
    try:
        print(f"DEBUG [PlotUnmet]: Plotting unmet demand for {city_name}...") 
        yearly_sums = results_df.groupby('year')[required_cols[1:]].sum()
        yearly_sums['total_demand_kwh'] = yearly_sums['unmet_demand_kwh'] + yearly_sums['energy_charged_this_hour_kwh']
        yearly_sums['unmet_pct'] = (yearly_sums['unmet_demand_kwh'] / yearly_sums['total_demand_kwh'].replace(0, np.nan)) * 100
        yearly_sums['unmet_pct'].fillna(0, inplace=True)
        if yearly_sums.empty: print(f"WARN [PlotUnmet]: No yearly unmet demand data found after grouping for {city_name}."); return None

        fig, ax = _setup_plot((8, 5), f'Yearly Unmet EV Charging Demand ({city_name})', 'Year', 'Unmet Demand (%)')
        ax.bar(yearly_sums.index, yearly_sums['unmet_pct'], color='orange', edgecolor='black')
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
        ax.grid(True, axis='y', linestyle=':')
        ax.set_ylim(0, max(10, yearly_sums['unmet_pct'].max() * 1.1) if yearly_sums['unmet_pct'].max() > 0 else 10) 
        ax.set_xticks(yearly_sums.index)
        ax.set_xticklabels(yearly_sums.index.astype(int))
        fig.tight_layout()
        return fig
    except Exception as e:
        print(f"ERROR [PlotUnmet]: Failed plotting unmet demand for {city_name}: {e}")
        traceback.print_exc()
        plt.close(fig) if 'fig' in locals() else None
        return None

if __name__ == "__main__":
    print("--- Starting EV Fleet Energy Balance Simulation (v3.6 - Combined Solar/Wind) ---")
    print("!!! NOTE: Ensure 'CITY_RENEWABLE_CAPACITY_PATH' CSV has separate solar/wind columns (profile IDs & MWs). !!!")
    print("!!! NOTE: Financial calculations use BASELINE costs in CONFIG & PLACEHOLDER city factors. Modify get_city_financial_factors() for real dynamic results. !!!")
    overall_start_time = timer.time()

    essential_paths = [ CONFIG['CITIES_CSV_PATH'], CONFIG['CITY_RENEWABLE_CAPACITY_PATH'], CONFIG['RENEWABLE_PROFILES_DATA_PATH'], CONFIG['PYBAMM_LOOKUP_PATH'] ]
    missing_paths = [str(p.resolve()) for p in essential_paths if not p.is_file()];
    if missing_paths: print(f"ERROR: Missing essential files:\n" + "\n".join(missing_paths)); sys.exit(1)
    try: load_pybamm_lookup()
    except Exception as e: print(f"CRITICAL ERROR loading PyBaMM lookup: {e}"); sys.exit(1)
    if not _load_and_clean_data(CONFIG): print("Exiting due to data loading errors."); sys.exit(1)

    cities_main = set(cities_df_global[CONFIG['CITY_COL']].unique()); cities_cap = set(city_renewable_capacity_df_global[CONFIG['RENEW_CITY_COL']].unique()); profiles_avail = set(renewable_profiles_data_df_global[CONFIG['PROFILE_ID_COL']].unique())
    simulatable_cities = []
    solar_profile_col = CONFIG['RENEW_SOLAR_PROFILE_ID_COL']; wind_profile_col = CONFIG['RENEW_WIND_PROFILE_ID_COL']; solar_mw_cols = CONFIG['RENEW_SOLAR_MW_YEAR_COLS']; wind_mw_cols = CONFIG['RENEW_WIND_MW_YEAR_COLS']
    for city in sorted(list(cities_main.intersection(cities_cap))):
        cap_row_match = city_renewable_capacity_df_global[city_renewable_capacity_df_global[CONFIG['RENEW_CITY_COL']] == city]
        if cap_row_match.empty: continue
        cap_row = cap_row_match.iloc[0]; solar_profile_id = cap_row.get(solar_profile_col, ""); wind_profile_id = cap_row.get(wind_profile_col, "")
        if not isinstance(solar_profile_id, str) or not isinstance(wind_profile_id, str): continue
        solar_profile_id = solar_profile_id.strip(); wind_profile_id = wind_profile_id.strip()
        profiles_ok = bool(solar_profile_id and wind_profile_id and solar_profile_id in profiles_avail and wind_profile_id in profiles_avail)
        has_some_solar_mw = any(pd.notna(cap_row.get(mw_col)) and cap_row.get(mw_col, -1) >= 0 for mw_col in solar_mw_cols)
        has_some_wind_mw = any(pd.notna(cap_row.get(mw_col)) and cap_row.get(mw_col, -1) >= 0 for mw_col in wind_mw_cols)
        capacity_ok = has_some_solar_mw or has_some_wind_mw
        if profiles_ok and capacity_ok: simulatable_cities.append(city)

    if not simulatable_cities: print("\nCRITICAL ERROR: No cities found that are simulatable."); sys.exit(1)
    if not FUZZY_AVAILABLE: print("\n(Optional: Install 'fuzzywuzzy[speedup]' for city name suggestions via pip)")

    mode = "";
    while mode not in ['1', '2']: mode = input("Select mode:\n  1: Simulate Single City\n  2: Compare Two Cities\nEnter choice (1 or 2): ").strip()

    try: global_config_tuple_for_cache = _create_global_config_tuple(CONFIG)
    except Exception as e: print(f"ERROR creating config tuple for cache: {e}"); sys.exit(1)

    results_store = {}; cities_to_run = []
    if mode == '1':
        city_name_selected = select_city(simulatable_cities, "(City 1)") 
        cities_to_run = [city_name_selected]
    elif mode == '2':
        city1 = select_city(simulatable_cities, "(City 1)") 
        city2 = "";
        while not city2 or city2 == city1:
             city2 = select_city(simulatable_cities, f"(City 2, different from {city1})") 
             if city2 == city1: print("Please select a city different from City 1.")
        cities_to_run = [city1, city2]
    else: print("Invalid mode selected. Exiting."); sys.exit(1)

    all_sims_ok = True 

    len_years = len(CONFIG.get('RENEW_CAPACITY_YEARS', []))
    len_solar_cols = len(CONFIG.get('RENEW_SOLAR_MW_YEAR_COLS', []))
    len_wind_cols = len(CONFIG.get('RENEW_WIND_MW_YEAR_COLS', []))
    if not (len_years == len_solar_cols == len_wind_cols):
        print("\nCRITICAL CONFIGURATION ERROR:")
        print(f"  Mismatch in derived configuration list lengths:")
        print(f"  - RENEW_CAPACITY_YEARS: {len_years}")
        print(f"  - RENEW_SOLAR_MW_YEAR_COLS: {len_solar_cols}")
        print(f"  - RENEW_WIND_MW_YEAR_COLS: {len_wind_cols}")
        print(f"  This likely indicates an issue with the 'RENEW_MW_YEAR_COLS_TUPLES' definition or parsing.")
        print("Exiting due to configuration error.")
        sys.exit(1)

    for i, city_name_run in enumerate(cities_to_run):
        print(f"\n===== ({i+1}/{len(cities_to_run)}) Preparing Simulation for: {city_name_run} =====")

        try: 
            city_row_match = cities_df_global[cities_df_global[CONFIG['CITY_COL']] == city_name_run]
            cap_row_match = city_renewable_capacity_df_global[city_renewable_capacity_df_global[CONFIG['RENEW_CITY_COL']] == city_name_run]

            if city_row_match.empty:
                print(f"ERROR: Could not find city data for '{city_name_run}' in the main city list (cities_df_global). Skipping.")
                all_sims_ok = False
                continue
            if cap_row_match.empty:
                 print(f"ERROR: Could not find capacity data for '{city_name_run}' in the capacity list (city_renewable_capacity_df_global). Skipping.")
                 all_sims_ok = False
                 continue

            city_row = city_row_match.iloc[0]
            cap_row = cap_row_match.iloc[0]

            avg_temp_selected = city_row[CONFIG['TEMP_COL']]
            population_selected = city_row[CONFIG['POP_COL']]
            if not isinstance(avg_temp_selected, (int, float, np.number)) or pd.isna(avg_temp_selected):
                 print(f"ERROR: Invalid non-numeric or NaN temperature ({avg_temp_selected}) found for '{city_name_run}'. Skipping.")
                 all_sims_ok = False
                 continue
            if not isinstance(population_selected, (int, float, np.number)) or pd.isna(population_selected) or population_selected <= 0:
                 print(f"ERROR: Invalid non-numeric, NaN, or non-positive population ({population_selected}) found for '{city_name_run}'. Skipping.")
                 all_sims_ok = False
                 continue
            population_selected = int(population_selected) 

            solar_profile_id_selected = cap_row.get(CONFIG['RENEW_SOLAR_PROFILE_ID_COL'], "").strip()
            wind_profile_id_selected = cap_row.get(CONFIG['RENEW_WIND_PROFILE_ID_COL'], "").strip()

            if not solar_profile_id_selected or not wind_profile_id_selected:
                print(f"CRITICAL ERROR: Pre-validated city '{city_name_run}' is missing profile ID(s) at simulation stage. Skipping.")
                all_sims_ok = False
                continue

        except KeyError as ke:
             print(f"ERROR: Missing expected column key '{ke}' while retrieving data for '{city_name_run}'. Check CONFIG column names. Skipping.")
             all_sims_ok = False
             continue
        except Exception as data_err:
             print(f"ERROR: Unexpected error retrieving data for '{city_name_run}': {data_err}. Skipping.")
             traceback.print_exc()
             all_sims_ok = False
             continue

        solar_capacity_points = {}
        wind_capacity_points = {}
        try:
            for yr, col_s, col_w in zip(CONFIG['RENEW_CAPACITY_YEARS'], CONFIG['RENEW_SOLAR_MW_YEAR_COLS'], CONFIG['RENEW_WIND_MW_YEAR_COLS']):
                mw_s = cap_row.get(col_s) 
                mw_w = cap_row.get(col_w)

                if pd.notna(mw_s):
                    try:
                        solar_capacity_points[yr] = max(0.0, float(mw_s))
                    except (ValueError, TypeError):
                        pass
                if pd.notna(mw_w):
                    try:
                        wind_capacity_points[yr] = max(0.0, float(mw_w))
                    except (ValueError, TypeError):
                        pass

            solar_capacity_points_tuple = tuple(sorted(solar_capacity_points.items()))
            wind_capacity_points_tuple = tuple(sorted(wind_capacity_points.items()))

            if not solar_capacity_points and not wind_capacity_points:
                print(f"CRITICAL ERROR: No valid MW capacity data could be processed for {city_name_run} from row data. Skipping.")
                all_sims_ok = False
                continue

        except Exception as cap_prep_err:
            print(f"ERROR: Unexpected error preparing capacity data for '{city_name_run}': {cap_prep_err}. Skipping.")
            traceback.print_exc()
            all_sims_ok = False
            continue

        print(f"  Input Params: Pop={population_selected:,.0f}, Temp={avg_temp_selected:.1f}°C")
        print(f"  Solar Profile: '{solar_profile_id_selected}', Capacity Points (Yr, MW): {solar_capacity_points}")
        print(f"  Wind Profile:  '{wind_profile_id_selected}', Capacity Points (Yr, MW): {wind_capacity_points}")

        summary, soh_df, results_df, annual_fin_df = {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        simulation_run_ok = False 
        try:
            call_start = timer.time()
            summary, soh_df, results_df, annual_fin_df = run_full_city_simulation(
                city_name=city_name_run,
                avg_temp_c=avg_temp_selected,
                population=population_selected, 
                city_solar_capacity_tuple=solar_capacity_points_tuple, 
                city_wind_capacity_tuple=wind_capacity_points_tuple,   
                solar_profile_id=solar_profile_id_selected,         
                wind_profile_id=wind_profile_id_selected,           
                global_config_tuple=global_config_tuple_for_cache   
            )
            call_duration = timer.time() - call_start
            print(f"--- Simulation function call returned in {call_duration:.2f} sec ---")

            if (summary and isinstance(summary, dict) and
                isinstance(soh_df, pd.DataFrame) and not soh_df.empty and
                isinstance(results_df, pd.DataFrame) and not results_df.empty and
                isinstance(annual_fin_df, pd.DataFrame)): 
                if 'Avg_SoH_Cap_Percent' not in soh_df.columns:
                     print("ERROR: Simulation result validation failed - 'Avg_SoH_Cap_Percent' missing from soh_df.")
                elif 'energy_charged_this_hour_kwh' not in results_df.columns:
                     print("ERROR: Simulation result validation failed - 'energy_charged_this_hour_kwh' missing from results_df.")
                else:
                     simulation_run_ok = True
                     print("--- Simulation results appear valid (passed basic checks). ---")
            else:
                print("ERROR: Simulation function returned invalid or empty core results (summary dict empty, or soh/results df empty/invalid).")
                if not summary or not isinstance(summary, dict): print("  - Summary failed check.")
                if not isinstance(soh_df, pd.DataFrame) or soh_df.empty: print("  - SOH DataFrame failed check.")
                if not isinstance(results_df, pd.DataFrame) or results_df.empty: print("  - Results DataFrame failed check.")
                if not isinstance(annual_fin_df, pd.DataFrame): print("  - Annual Fin DataFrame failed type check.")


        except Exception as main_error:
            print(f"\n!!!!!! SIMULATION FAILED UNEXPECTEDLY for {city_name_run} during function call: {type(main_error).__name__}: {main_error} !!!!!!")
            traceback.print_exc()

        if simulation_run_ok:
            results_store[city_name_run] = {
                'summary': summary,
                'soh_df': soh_df,
                'results_df': results_df,
                'annual_fin_df': annual_fin_df
            }
            print(f"--- Results for {city_name_run} stored successfully. ---")
        else:
            all_sims_ok = False 
            print(f"--- Simulation for {city_name_run} FAILED or produced invalid results. Results not stored. ---")

    def format_value(x):
        if isinstance(x, (float, np.floating)): return f"{x:,.2f}" if abs(x) < 1e6 and abs(x) > 1e-3 and x!=0 else f"{x:.3g}"
        elif isinstance(x, (int, np.integer)): return f"{x:,}"
        else: return str(x)

    if not results_store: print("\n--- No simulation results available. Skipping post-processing and saving. ---")
    elif mode == '1':
        if all_sims_ok and len(cities_to_run) == 1:
            city_name = cities_to_run[0]
            if city_name not in results_store: print(f"ERROR: Results for {city_name} not found in store despite success flag. Skipping.")
            else:
                res = results_store[city_name]
                summary_to_print = res.get('summary', {})
                print("\n----- Simulation Results Summary -----")
                try:
                    print(pd.DataFrame([summary_to_print]).T.map(format_value))
                except Exception as fmt_err:
                     print(f"WARN: Error formatting summary: {fmt_err}. Printing raw summary.")
                     print(pd.DataFrame([summary_to_print]).T)

                print("\n!!! Reminder: Financial metrics depend on BASELINE costs & PLACEHOLDER city factors. !!!")
                print("\n----- Note on Charging Time -----"); base_cap_kwh = CONFIG.get('BATTERY_NOMINAL_CAPACITY_KWH', 60); example_charger_power = 50.0; charge_time_h = base_cap_kwh / example_charger_power if example_charger_power > 0 else float('inf'); print(f"Time(h) ≈ Batt_Cap(kWh) / Avg_Charger_Power(kW). Ex: {base_cap_kwh}kWh / {example_charger_power}kW ≈ {charge_time_h:.2f}h (Ignoring efficiency/SoH)"); print("-------------------------------\n")

                print("INFO: Generating plots...")
                plots_to_show = []
                try:
                    plot_definitions = [
                        (plot_soh_forecast, res.get('soh_df'), "SoH Forecast"),
                        (plot_energy_balance_sample, res.get('results_df'), "Energy Balance Sample"),
                        (plot_yearly_peaks, res.get('results_df'), "Yearly Peak Grid Draw"),
                        (plot_yearly_energy_contribution, res.get('results_df'), "Yearly Energy Contribution"),
                        (plot_yearly_renewable_utilization, res.get('results_df'), "Yearly Renewable Utilization"),
                        (plot_average_daily_charging_profile, res.get('results_df'), "Average Daily Charging Profile"),
                        (plot_yearly_unmet_demand, res.get('results_df'), "Yearly Unmet Demand"),
                        (plot_cumulative_financials, res.get('annual_fin_df'), "Cumulative Financials"),
                        (plot_annual_costs, res.get('annual_fin_df'), "Annual Costs"), 
                        (plot_annual_savings, res.get('annual_fin_df'), "Annual Savings")
                    ]
                except NameError as ne:
                     print(f"\nCRITICAL PLOTTING ERROR: Plotting function '{ne.name}' is not defined.")
                     print("Ensure all plot_... functions are defined before the if __name__ == '__main__': block.")
                     plot_definitions = [] 

                for plot_func, data_arg, name in plot_definitions:
                     if isinstance(data_arg, pd.DataFrame) and not data_arg.empty:
                         print(f"  Attempting plot: {name}"); fig = None
                         try:
                             fig = plot_func(data_arg, city_name=city_name, config=CONFIG)
                         except Exception as plot_err:
                             print(f"ERROR generating plot '{name}': {plot_err}"); traceback.print_exc()
                         if fig:
                             plots_to_show.append(fig)
                     else:
                         data_info = f"Type: {type(data_arg)}" if not isinstance(data_arg, pd.DataFrame) else "Empty DataFrame"
                         print(f"  Skipping plot '{name}' (Data invalid or empty: {data_info}).")

                if plots_to_show: print(f"\nINFO: Generated {len(plots_to_show)} plots. Displaying now..."); plt.show(block=True)
                else: print("WARN: No plots were generated.")

                _save_aggregate_summary(res.get('summary', {}), city_name, CONFIG)
                _save_hourly_results(res.get('results_df'), city_name, CONFIG)
        else: print("\n--- Simulation failed. Skipping results display, plotting, and saving. ---")

    elif mode == '2':
        if all_sims_ok and len(cities_to_run) == 2 and all(c in results_store for c in cities_to_run):
            city1, city2 = cities_to_run[0], cities_to_run[1]; res1, res2 = results_store[city1], results_store[city2]
            print(f"\n----- Comparison Summary: {city1} vs {city2} -----")
            summary1_to_print = res1.get('summary', {})
            summary2_to_print = res2.get('summary', {})
            try:
                print(f"\nSummary for {city1}:"); print(pd.DataFrame([summary1_to_print]).T.map(format_value))
                print(f"\nSummary for {city2}:"); print(pd.DataFrame([summary2_to_print]).T.map(format_value))
            except Exception as fmt_err:
                 print(f"WARN: Error formatting summary: {fmt_err}. Printing raw summaries.")
                 print(f"\nSummary for {city1}:"); print(pd.DataFrame([summary1_to_print]).T)
                 print(f"\nSummary for {city2}:"); print(pd.DataFrame([summary2_to_print]).T)

            print("\n!!! Reminder: Financial metrics depend on BASELINE costs & PLACEHOLDER city factors. !!!")
            print("\nINFO: Generating comparison plots..."); comparison_plots = []
            try:
                comparison_plots = plot_city_comparison(res1.get('summary', {}), res2.get('summary', {}), res1.get('annual_fin_df'), res2.get('annual_fin_df'), city1, city2, CONFIG)
            except NameError as ne:
                 print(f"\nCRITICAL PLOTTING ERROR: Comparison plotting function '{ne.name}' is not defined.")
                 print("Ensure plot_city_comparison function is defined before the if __name__ == '__main__': block.")
            except Exception as comp_plot_err: print(f"ERROR generating comparison plots: {comp_plot_err}"); traceback.print_exc()
            if comparison_plots: print(f"INFO: Generated {len(comparison_plots)} comparison plots. Displaying now..."); plt.show(block=True)
            else: print("WARN: No comparison plots were generated.")

            print(f"INFO: Saving results for {city1}..."); _save_aggregate_summary(res1.get('summary', {}), city1, CONFIG); _save_hourly_results(res1.get('results_df'), city1, CONFIG)
            print(f"INFO: Saving results for {city2}..."); _save_aggregate_summary(res2.get('summary', {}), city2, CONFIG); _save_hourly_results(res2.get('results_df'), city2, CONFIG)
        else: print("\n--- One or more simulations failed or produced invalid results. Skipping comparison and saving. ---")

    overall_elapsed = timer.time() - overall_start_time; cache_info_str = "N/A";
    try: info = run_full_city_simulation.cache_info(); cache_info_str = f"Hits={info.hits}, Misses={info.misses}, Size={info.currsize}/{info.maxsize}"
    except Exception as e_cache: cache_info_str = f"Error retrieving info: {e_cache}"
    print(f"\n--- Run Complete ({overall_elapsed:.2f} seconds) ---"); print(f"--- Cache Info: {cache_info_str} ---"); print("="*60)