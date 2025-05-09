import pybamm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time as timer
from pathlib import Path
import sys
import traceback
from collections.abc import Iterable
import functools


try:
    from fuzzywuzzy import process as fuzzy_process
    HAS_FUZZYWUZZY = True
except ImportError:
    HAS_FUZZYWUZZY = False



MODEL_DIR = Path("C:/Users/snsur/Desktop/ster/")
MODEL_FILENAME = "my_actual_degradation_model.py"
MODEL_FUNCTION_NAME = "my_actual_degradation_model"
model_dir_str = str(MODEL_DIR.resolve())
my_actual_degradation_model_func = None
USE_ACTUAL_MODEL = False
if MODEL_DIR.is_dir():
    if model_dir_str not in sys.path: sys.path.append(model_dir_str)
    try:
        module_name = MODEL_FILENAME.replace(".py", "")
        degradation_module = __import__(module_name)
        my_actual_degradation_model_func = getattr(degradation_module, MODEL_FUNCTION_NAME)
        USE_ACTUAL_MODEL = True
        print(f"Imported custom degradation model: '{MODEL_FUNCTION_NAME}'.")
    except ModuleNotFoundError:
        print(f"WARN: Custom model file '{MODEL_FILENAME}' not found in '{model_dir_str}'. Using placeholder.")
        USE_ACTUAL_MODEL = False
    except AttributeError:
        print(f"WARN: Function '{MODEL_FUNCTION_NAME}' not found in '{MODEL_FILENAME}'. Using placeholder.")
        USE_ACTUAL_MODEL = False
    except Exception as import_err:
        print(f"WARN: Failed importing custom model or function ({import_err}). Using placeholder.")
        traceback.print_exc()
        USE_ACTUAL_MODEL = False
else:
    print(f"WARN: Custom model directory '{MODEL_DIR}' not found. Using placeholder.")
    USE_ACTUAL_MODEL = False



CITIES_CSV_PATH = Path("C:/Users/snsur/Desktop/ster/city_temperatures_clean.csv")
CITY_RENEWABLE_DATA_PATH = Path("C:/Users/snsur/Desktop/ster/city_renewable_supply_enhanced.csv")
PYBAMM_LOOKUP_PATH = Path("./pybamm_temperature_lookup.csv")


CITY_COL = 'city_name'; TEMP_COL = 'avg_temp_celsius'; POP_COL = 'population'
CITY_RENEW_CITY_COL = 'city_name'; CITY_RENEW_POWER_COL = 'power_kw_per_mw'


EV_ADOPTION_RATE_PER_CAPITA = 0.01; RENEWABLE_MW_PER_MILLION_POP = 10.0
MIN_INITIAL_FLEET_SIZE = 10; MIN_INITIAL_RENEWABLE_MW = 1.0
FORECAST_YEARS = 10; FLEET_GROWTH_RATE = 0.15; RENEWABLE_GROWTH_RATE = 0.10


AVG_DAILY_KM_PER_EV = 40; BASE_EFFICIENCY_KWH_KM = 0.18; CYCLES_PER_DAY = 1.0
INITIAL_AVG_SOH_PERCENT = 100.0; INITIAL_AVG_RESISTANCE_PERCENT = 100.0
BATTERY_NOMINAL_CAPACITY_KWH = 60; CHEMISTRY = "Li-ion"; TRAFFIC_LEVEL = 1; USER_PROFILE = "Average User"


PYBAMM_PARAM_SET = "Chen2020"; PYBAMM_SIM_TYPE = 'charge'; PYBAMM_SIM_CRATE = 1.5
HEAT_TRANSFER_COEFF = 25; PYBAMM_INITIAL_SOC = 0.1


CHARGER_MIX = {'L2':{'power':7.0,'fraction':0.60},'DCFC_50':{'power':50.0,'fraction':0.25},'DCFC_150':{'power':150.0,'fraction':0.15}}
RESISTANCE_IMPACT_FACTOR = 0.2; CHARGE_START_HOUR = 0; CHARGE_END_HOUR = 4; SMART_CHARGING_STRATEGY = 'unmanaged'
USE_GRID_STORAGE = True; STORAGE_CAPACITY_KWH = 20000
STORAGE_MAX_CHARGE_KW = 5000; STORAGE_MAX_DISCHARGE_KW = 5000; STORAGE_EFFICIENCY = 0.85
INITIAL_MAX_GRID_POWER_KW = 50000
GRID_LIMIT_GROWTH_RATE = 0.10


SUMMARY_OUTPUT_FILENAME_BASE = "simulation_summary"



_pybamm_lookup_table = None




def load_pybamm_lookup():
    """Loads the pre-computed PyBaMM temperature lookup table."""
    global _pybamm_lookup_table
    if _pybamm_lookup_table is None:
        if PYBAMM_LOOKUP_PATH.is_file():
            print(f"INFO: Loading PyBaMM lookup table from: {PYBAMM_LOOKUP_PATH}")
            try:
                _pybamm_lookup_table = pd.read_csv(PYBAMM_LOOKUP_PATH)
                _pybamm_lookup_table = _pybamm_lookup_table.sort_values(by="ambient_temp_c").reset_index(drop=True)
                if _pybamm_lookup_table[['avg_batt_temp_c', 'peak_batt_temp_c']].isnull().values.any():
                    print("WARN: NaNs found in PyBaMM lookup table! Interpolation might fail.")
                print("INFO: PyBaMM lookup table loaded successfully.")
            except Exception as e:
                 print(f"ERROR: Failed to load or process PyBaMM lookup table: {e}")
                 _pybamm_lookup_table = pd.DataFrame()
                 raise
        else:
            print(f"CRITICAL ERROR: PyBaMM lookup file not found at {PYBAMM_LOOKUP_PATH}.")
            print("Please run the pre-computation script ('create_lookup_table.py') first.")
            _pybamm_lookup_table = pd.DataFrame()
            raise FileNotFoundError(f"PyBaMM lookup file not found: {PYBAMM_LOOKUP_PATH}")

    if _pybamm_lookup_table.empty:
        raise RuntimeError("PyBaMM lookup table failed to load or is empty.")
    return _pybamm_lookup_table


def run_pybamm_simulation(ambient_temp_c, c_rate=None, h_coeff=None, initial_soc=None, sim_type=None, param_set=None, custom_params=None):
    """
    Retrieves average and peak battery temperatures from a pre-computed lookup table
    based on ambient temperature using interpolation.

    NOTE: Other parameters (c_rate, h_coeff, etc.) are now ignored as the lookup
          table is assumed to be generated with the standard config values.
    """
    if np.isnan(ambient_temp_c): print("ERROR: Ambient temp NaN."); return np.nan, np.nan

    try:
        lookup_df = load_pybamm_lookup()


        interp_avg_t = np.interp(ambient_temp_c, lookup_df['ambient_temp_c'], lookup_df['avg_batt_temp_c'])
        interp_peak_t = np.interp(ambient_temp_c, lookup_df['ambient_temp_c'], lookup_df['peak_batt_temp_c'])


        min_lookup_t = lookup_df['ambient_temp_c'].min()
        max_lookup_t = lookup_df['ambient_temp_c'].max()
        if not (min_lookup_t <= ambient_temp_c <= max_lookup_t):
             print(f"  WARN: Ambient temp {ambient_temp_c:.1f}C is outside pre-computed range [{min_lookup_t:.1f}C, {max_lookup_t:.1f}C]. Results are extrapolated.")


        if np.isnan(interp_avg_t) or np.isnan(interp_peak_t):
             print("ERROR: Interpolation resulted in NaN. Check lookup table content.")
             return np.nan, np.nan
        return interp_avg_t, interp_peak_t

    except (FileNotFoundError, RuntimeError) as e:
        print(f"  {e}")
        print("  Cannot proceed without valid PyBaMM lookup table.")
        return np.nan, np.nan
    except Exception as e:
         print(f"ERROR during PyBaMM lookup/interpolation: {type(e).__name__}: {e}")
         traceback.print_exc()
         return np.nan, np.nan


def predict_degradation_wrapper(initial_soh_percent, resistance_initial_percent, cycles, avg_temp_c, peak_temp_c, traffic_level, user_profile, chemistry, **kwargs):
    if np.isnan(avg_temp_c) or np.isnan(peak_temp_c): print("WARN: NaN temp -> 0 degradation."); return 0.0, 0.0
    if USE_ACTUAL_MODEL and my_actual_degradation_model_func is not None:
        try:
            fade_delta, res_delta = my_actual_degradation_model_func(start_soh_capacity=initial_soh_percent, start_soh_resistance=resistance_initial_percent, num_cycles=cycles, avg_temperature_c=avg_temp_c, peak_charge_temperature_c=peak_temp_c, traffic_condition=traffic_level, usage_pattern=user_profile, cell_chemistry=chemistry, **kwargs)
            fade_delta=float(fade_delta) if pd.notna(fade_delta) else 0.0
            res_delta=float(res_delta) if pd.notna(res_delta) else 0.0
            if res_delta<0: print(f"WARN: Neg res increase ({res_delta:.3f}%). Setting to 0."); res_delta=0.0
            if fade_delta<0: print(f"WARN: Neg fade ({fade_delta:.3f}%). Setting to 0."); fade_delta=0.0
            return fade_delta, res_delta
        except Exception as e: print(f"ERROR calling imported model: {e}. Falling back."); traceback.print_exc();

    b_fade=0.00015 if chemistry=="Li-ion" else 0.00020; b_res=0.00025 if chemistry=="Li-ion" else 0.00020
    t_fact = 1 + (max(0, avg_temp_c - 25) / 15)**1.5
    tr_fact = 1 + (traffic_level * 0.1)
    u_fact = 1.5 if user_profile == "Taxi/Ride Share" else (0.8 if user_profile == "Gentle Commuter" else 1.0)
    s_cap = max(0.5, initial_soh_percent / 100.0); s_res = 1.0
    cap_d = cycles * b_fade * t_fact * tr_fact * u_fact * s_cap * 100
    res_d = cycles * b_res * t_fact * tr_fact * u_fact * s_res * 100
    return min(cap_d, initial_soh_percent), max(0, res_d)


def forecast_fleet_degradation(years, initial_soh_cap, initial_soh_res_percent, cycles_per_day, avg_temp_c, peak_temp_c, traffic_level, user_profile, chemistry):
    print(f"\nForecasting degradation for {years} years...");
    if np.isnan(avg_temp_c) or np.isnan(peak_temp_c): print("ERROR: NaN temp input to forecast_fleet_degradation."); return pd.DataFrame(columns=["Year", "Avg_SoH_Cap_Percent", "Resistance_Percent"])
    history=[]; current_soh_cap=initial_soh_cap; current_res_pct=initial_soh_res_percent; cycles_yr=cycles_per_day*365
    history.append({"Year":0, "Avg_SoH_Cap_Percent":current_soh_cap, "Resistance_Percent":current_res_pct})
    for year in range(1, years+1):
        if current_soh_cap<=20: print(f"INFO: Year {year}: Avg SoH Cap reached EoL threshold (<=20%). Halting degradation."); history.append({"Year":year,"Avg_SoH_Cap_Percent":current_soh_cap,"Resistance_Percent":current_res_pct}); break
        fade_delta, res_delta = predict_degradation_wrapper(initial_soh_percent=current_soh_cap, resistance_initial_percent=current_res_pct, cycles=cycles_yr, avg_temp_c=avg_temp_c, peak_temp_c=peak_temp_c, traffic_level=traffic_level, user_profile=user_profile, chemistry=chemistry)
        if np.isnan(fade_delta) or np.isnan(res_delta):
            print(f"WARN: NaN degradation predicted in Year {year}. Keeping SoH constant for this year.");
            history.append({"Year":year,"Avg_SoH_Cap_Percent":current_soh_cap,"Resistance_Percent":current_res_pct})
        else:
             current_soh_cap-=fade_delta; current_res_pct+=res_delta
             current_soh_cap=max(0,current_soh_cap); current_res_pct=max(INITIAL_AVG_RESISTANCE_PERCENT, current_res_pct)
             history.append({"Year":year,"Avg_SoH_Cap_Percent":current_soh_cap,"Resistance_Percent":current_res_pct})

    last_year = history[-1]["Year"]; last_cap=history[-1]["Avg_SoH_Cap_Percent"]; last_res=history[-1]["Resistance_Percent"]
    for year in range(last_year + 1, years + 1): history.append({"Year":year, "Avg_SoH_Cap_Percent":last_cap, "Resistance_Percent":last_res})
    print(f"  Degradation forecast complete. Final State (Year {years}): Cap SoH={current_soh_cap:.2f}%, Res={current_res_pct:.2f}%"); return pd.DataFrame(history)


def load_renewable_supply_from_city_data(city_renewable_df, current_city_name, city_match_col, power_val_col, start_date, end_date, initial_capacity_mw, growth_rate, forecast_years):
    print(f"\nLooking up renewable generation potential for: {current_city_name}"); print("INFO: Using constant hourly power output per MW capacity (based on input file).")
    city_data=city_renewable_df[city_renewable_df[city_match_col]==current_city_name]
    if city_data.empty: raise ValueError(f"City '{current_city_name}' missing from renewable supply data file.")
    if len(city_data)>1: print(f"WARN: Multiple entries for '{current_city_name}' in renewable data. Using first entry.");
    const_power_kw_per_mw = city_data[power_val_col].iloc[0]
    if pd.isna(const_power_kw_per_mw): raise ValueError(f"Invalid (NaN) power value for {current_city_name} in renewable data.")
    print(f"  Found value: {const_power_kw_per_mw:.2f} kW/MW"); const_power_kw_per_mw = max(0, const_power_kw_per_mw)
    full_end_date=pd.Timestamp(end_date)+pd.Timedelta(hours=23); hourly_index=pd.date_range(start=start_date,end=full_end_date,freq='h')
    supply_df=pd.DataFrame(index=hourly_index); supply_df['power_kw_per_mw']=const_power_kw_per_mw
    supply_df['year']=supply_df.index.year; start_year=start_date.year
    supply_df['capacity_mw']=initial_capacity_mw*((1+growth_rate)**(supply_df['year']-start_year))
    supply_df['supply_kw']=supply_df['power_kw_per_mw']*supply_df['capacity_mw']
    print(f"  Renewable supply calculation complete. (Yr 1: {initial_capacity_mw:.1f} MW, Yr {forecast_years}: {supply_df['capacity_mw'].iloc[-1]:.1f} MW)"); return supply_df[['supply_kw']].fillna(0)


def simulate_energy_balance_and_charging(soh_forecast_df, fleet_size_initial, growth_rate, daily_km, efficiency, nominal_capacity, supply_df, charger_mix, resistance_impact_factor, smart_charging_strategy, charge_start_hour, charge_end_hour,
                                         initial_max_grid_power_kw, grid_limit_growth_rate,
                                         use_grid_storage, storage_capacity_kwh, storage_max_charge_kw, storage_max_discharge_kw, storage_efficiency, forecast_years):
    print("\nSimulating Hourly Energy Balance (Dynamic Grid Limit & Smoothed Demand)..."); start_balance = timer.time()
    start_date=supply_df.index.min(); hourly_index=supply_df.index


    num_hours_total = len(hourly_index)
    chg_hrs = []
    chg_hrs_indices = {}
    if charge_start_hour == charge_end_hour:
        chg_hrs = list(range(0, 24))
        print("  Charge window: 24 hours (00:00 - 23:59)")
    elif charge_start_hour < charge_end_hour:
        chg_hrs = list(range(charge_start_hour, charge_end_hour))
        print(f"  Charge window: {charge_start_hour:02d}:00 - {charge_end_hour-1:02d}:59")
    else:
        chg_hrs = list(range(charge_start_hour, 24)) + list(range(0, charge_end_hour))
        print(f"  Charge window: {charge_start_hour:02d}:00 - 23:59 and 00:00 - {charge_end_hour-1:02d}:59")

    for idx, hour_val in enumerate(chg_hrs): chg_hrs_indices[hour_val] = idx
    num_charge_hours_per_day = len(chg_hrs)
    if num_charge_hours_per_day == 0: print("WARN: Charge window has zero hours!")


    results_df = pd.DataFrame(index=hourly_index)
    cols = ['fleet_size', 'avg_soh_cap', 'resistance_pct', 'supply_kw', 'max_grid_power_kw',
            'unused_renewables_kw', 'total_hourly_potential_charge_need_kw',
            'energy_charged_this_hour_kwh', 'demand_met_by_renewables_kwh',
            'demand_met_by_storage_kwh', 'demand_met_by_grid_kwh',
            'charge_to_storage_from_renewables_kwh', 'charge_to_storage_from_grid_kwh',
            'unmet_demand_kwh', 'grid_draw_kw', 'storage_soc_kwh']
    for c in cols: results_df[c] = 0.0
    results_df['supply_kw'] = supply_df['supply_kw']
    results_df['unused_renewables_kw'] = supply_df['supply_kw']


    daily_idx = pd.date_range(start=start_date, end=hourly_index.max().normalize(), freq='D')
    daily_state = pd.DataFrame(index=daily_idx)
    daily_state['fleet_size'] = 0.0
    daily_state['avg_soh_cap'] = INITIAL_AVG_SOH_PERCENT
    daily_state['resistance_pct'] = INITIAL_AVG_RESISTANCE_PERCENT

    current_year_fleet_size = fleet_size_initial
    if 'Year' not in soh_forecast_df.columns or 'Avg_SoH_Cap_Percent' not in soh_forecast_df.columns or 'Resistance_Percent' not in soh_forecast_df.columns:
        raise ValueError("SoH forecast missing required columns ('Year', 'Avg_SoH_Cap_Percent', 'Resistance_Percent').")
    soh_idx = soh_forecast_df.set_index('Year')


    years = range(start_date.year, start_date.year + forecast_years)
    yearly_max_grid_power = {
        yr: initial_max_grid_power_kw * ((1 + grid_limit_growth_rate) ** (yr - start_date.year))
        for yr in years
    }
    print(f"  Dynamic Grid Limit: Starts {initial_max_grid_power_kw/1000:.1f} MW, grows at {grid_limit_growth_rate*100:.1f}%/year")

    for yr_offset in range(forecast_years):
        current_sim_year = start_date.year + yr_offset
        soh_c = INITIAL_AVG_SOH_PERCENT; res_p = INITIAL_AVG_RESISTANCE_PERCENT
        try:
            soh_c = soh_idx.loc[yr_offset, 'Avg_SoH_Cap_Percent']
            res_p = soh_idx.loc[yr_offset, 'Resistance_Percent']
        except KeyError:
            print(f"WARN: SoH data missing for forecast Year offset {yr_offset} (Sim Year {current_sim_year}). Attempting fallback.")
            if yr_offset > 0 and (yr_offset - 1) in soh_idx.index:
                soh_c = soh_idx.loc[yr_offset - 1, 'Avg_SoH_Cap_Percent']
                res_p = soh_idx.loc[yr_offset - 1, 'Resistance_Percent']


        yr_s = pd.Timestamp(f"{current_sim_year}-01-01"); yr_e = pd.Timestamp(f"{current_sim_year}-12-31")
        dates_in_year = daily_state.index[(daily_state.index >= yr_s) & (daily_state.index <= yr_e)]
        if not dates_in_year.empty:
            daily_state.loc[dates_in_year, ['fleet_size', 'avg_soh_cap', 'resistance_pct']] = [current_year_fleet_size, soh_c, res_p]
        current_year_fleet_size *= (1 + growth_rate)

    daily_kwh_ev = daily_km * efficiency
    daily_state['total_daily_fleet_kwh_need'] = daily_state['fleet_size'] * daily_kwh_ev
    daily_state = daily_state.ffill().fillna(0)


    stor_soc = 0.0
    avg_chg_kw = sum(m['fraction'] * m['power'] for m in charger_mix.values())
    remaining_daily_need_kwh = 0.0


    eff_inv = 1.0 / storage_efficiency if storage_efficiency > 0 else float('inf')


    loop_start_time = timer.time()
    results_list = []

    for dt, row in daily_state.iterrows():
        current_day = dt.normalize()
        current_year = current_day.year

        f_s, soh_c, res_p = row['fleet_size'], row['avg_soh_cap'], row['resistance_pct']
        daily_total_need = row['total_daily_fleet_kwh_need']
        remaining_daily_need_kwh = daily_total_need
        current_max_grid_power_kw = yearly_max_grid_power.get(current_year, initial_max_grid_power_kw)


        for hr in range(24):
            current_timestamp = current_day + pd.Timedelta(hours=hr)
            if current_timestamp not in hourly_index: continue

            hourly_result = { 'timestamp': current_timestamp, 'fleet_size': f_s, 'avg_soh_cap': soh_c, 'resistance_pct': res_p }
            hourly_result['max_grid_power_kw'] = current_max_grid_power_kw

            available_renewables_kw = supply_df.loc[current_timestamp, 'supply_kw']
            hourly_result['supply_kw'] = available_renewables_kw


            target_charge_power_kw = 0.0
            if hr in chg_hrs and remaining_daily_need_kwh > 1e-6:

                current_hour_index = chg_hrs_indices.get(hr, -1)
                hours_left_in_window = max(1, num_charge_hours_per_day - current_hour_index) if current_hour_index != -1 else 1

                ideal_power_this_hour = remaining_daily_need_kwh / hours_left_in_window

                resistance_increase_factor = max(0, (res_p - 100.0) / 100.0)
                resistance_penalty = 1.0 + (resistance_increase_factor * resistance_impact_factor)
                effective_avg_charger_power = avg_chg_kw / resistance_penalty if resistance_penalty > 0 else 0
                max_potential_fleet_draw_kw = f_s * effective_avg_charger_power


                target_charge_power_kw = max(0, min(max_potential_fleet_draw_kw, ideal_power_this_hour))

            hourly_result['total_hourly_potential_charge_need_kw'] = target_charge_power_kw


            power_ev_from_renewables = 0.0
            power_storage_from_renewables = 0.0
            power_ev_from_storage = 0.0
            power_ev_from_grid = 0.0
            power_storage_from_grid = 0.0

            remaining_ev_demand_kw = target_charge_power_kw


            power_ev_from_renewables = min(remaining_ev_demand_kw, available_renewables_kw)
            remaining_ev_demand_kw -= power_ev_from_renewables
            available_renewables_kw -= power_ev_from_renewables


            if use_grid_storage:
                space_kwh = storage_capacity_kwh - stor_soc
                storage_charge_limit_by_rate_kw = storage_max_charge_kw
                storage_charge_limit_by_space_kw = max(0, space_kwh * eff_inv)
                storage_charge_potential_kw = min(storage_charge_limit_by_rate_kw, storage_charge_limit_by_space_kw)

                power_storage_from_renewables = min(available_renewables_kw, storage_charge_potential_kw)
                if power_storage_from_renewables > 1e-6:
                    stor_soc += power_storage_from_renewables * storage_efficiency
                    available_renewables_kw -= power_storage_from_renewables
                    stor_soc = min(stor_soc, storage_capacity_kwh)

            hourly_result['unused_renewables_kw'] = max(0, available_renewables_kw)


            if use_grid_storage and remaining_ev_demand_kw > 0:
                discharge_limit_by_rate_kw = storage_max_discharge_kw
                discharge_limit_by_soc_kw = max(0, stor_soc)
                storage_discharge_potential_kw = min(discharge_limit_by_rate_kw, discharge_limit_by_soc_kw)

                power_ev_from_storage = min(remaining_ev_demand_kw, storage_discharge_potential_kw)
                if power_ev_from_storage > 1e-6:
                    stor_soc -= power_ev_from_storage
                    remaining_ev_demand_kw -= power_ev_from_storage
                    stor_soc = max(0, stor_soc)


            grid_draw_for_ev_kw = 0.0
            if remaining_ev_demand_kw > 0:
                grid_available_for_ev = current_max_grid_power_kw
                power_ev_from_grid = min(remaining_ev_demand_kw, grid_available_for_ev)
                remaining_ev_demand_kw -= power_ev_from_grid
                grid_draw_for_ev_kw = power_ev_from_grid


            grid_avail_after_ev_kw = max(0, current_max_grid_power_kw - grid_draw_for_ev_kw)
            if use_grid_storage and hr in chg_hrs and grid_avail_after_ev_kw > 0:
                space_kwh = storage_capacity_kwh - stor_soc
                storage_charge_limit_by_rate_kw = storage_max_charge_kw
                storage_charge_limit_by_space_kw = max(0, space_kwh * eff_inv)
                storage_charge_potential_kw = min(storage_charge_limit_by_rate_kw, storage_charge_limit_by_space_kw)

                power_storage_from_grid = min(grid_avail_after_ev_kw, storage_charge_potential_kw)

                if power_storage_from_grid > 1e-6:
                    stor_soc += power_storage_from_grid * storage_efficiency
                    stor_soc = min(stor_soc, storage_capacity_kwh)


            total_grid_draw_kw = grid_draw_for_ev_kw + power_storage_from_grid
            total_charged_this_hour_kwh = power_ev_from_renewables + power_ev_from_storage + power_ev_from_grid
            unmet_demand_this_hour_kwh = max(0, remaining_ev_demand_kw)


            remaining_daily_need_kwh -= total_charged_this_hour_kwh
            remaining_daily_need_kwh = max(0, remaining_daily_need_kwh)


            hourly_result['energy_charged_this_hour_kwh'] = total_charged_this_hour_kwh
            hourly_result['demand_met_by_renewables_kwh'] = power_ev_from_renewables
            hourly_result['demand_met_by_storage_kwh'] = power_ev_from_storage
            hourly_result['demand_met_by_grid_kwh'] = power_ev_from_grid
            hourly_result['charge_to_storage_from_renewables_kwh'] = power_storage_from_renewables
            hourly_result['charge_to_storage_from_grid_kwh'] = power_storage_from_grid
            hourly_result['grid_draw_kw'] = total_grid_draw_kw
            hourly_result['unmet_demand_kwh'] = unmet_demand_this_hour_kwh
            if use_grid_storage:
                hourly_result['storage_soc_kwh'] = stor_soc
            else:
                 hourly_result['storage_soc_kwh'] = 0.0

            results_list.append(hourly_result)


    results_df = pd.DataFrame(results_list).set_index('timestamp')

    loop_end_time = timer.time()
    print(f"  Hourly energy balance loop took {loop_end_time - loop_start_time:.2f} sec.")
    print("  Hourly energy balance simulation complete."); return results_df.fillna(0)


@functools.lru_cache(maxsize=16)
def run_full_city_simulation(city_name, avg_temp_c, population, config_tuple):
    """
    Orchestrates the full simulation for a city, designed to be cached.
    Takes city identifiers and a tuple of relevant config parameters.
    Returns a tuple: (summary_dict, soh_history_df, results_df) or raises an error.
    """
    sim_start_time = timer.time()
    print(f"\n------ CACHE MISS: Running Full Simulation for {city_name} ------")


    ( forecast_years, fleet_growth_rate, renewable_growth_rate, avg_daily_km,
      base_efficiency, cycles_per_day, chemistry, traffic_level, user_profile,
      pybamm_param_set_ref, pybamm_sim_type_ref, pybamm_crate_ref, heat_coeff_ref, pybamm_init_soc_ref,
      charger_config_tuple_of_tuples,
      resistance_impact, charge_start, charge_end, smart_strategy,
      use_storage, storage_cap, storage_chg_kw, storage_dis_kw, storage_eff,
      initial_soh_cap, initial_soh_res, batt_nom_cap, renew_power_col,
      renew_city_col,
      initial_max_grid_power_kw_cfg, grid_limit_growth_rate_cfg
    ) = config_tuple


    local_charger_mix = {}
    try:
        for charger_name, inner_details_tuple in charger_config_tuple_of_tuples:
            local_charger_mix[charger_name] = dict(inner_details_tuple)
    except Exception as e:
        print(f"ERROR reconstructing charger mix from cache tuple: {e}")
        raise ValueError("Invalid charger mix configuration passed in cache tuple.") from e


    try:

        initial_fleet_size = max(MIN_INITIAL_FLEET_SIZE, int(population*EV_ADOPTION_RATE_PER_CAPITA))
        initial_renewable_mw = max(MIN_INITIAL_RENEWABLE_MW, (population/1e6)*RENEWABLE_MW_PER_MILLION_POP)
        print(f"  Initial Fleet Size (Year 1): {initial_fleet_size:,d} EVs")
        print(f"  Initial Renewable Capacity (Year 1): {initial_renewable_mw:.1f} MW")
        sim_start_date = pd.Timestamp('2025-01-01');
        sim_end_date = sim_start_date + pd.DateOffset(years=forecast_years, days=-1)
        print(f"  Simulation Period: {sim_start_date.date()} to {sim_end_date.date()} ({forecast_years} years)")


        rep_avg_temp, rep_peak_temp = run_pybamm_simulation(ambient_temp_c=avg_temp_c)
        if pd.isna(rep_avg_temp) or pd.isna(rep_peak_temp):
             raise ValueError(f"PyBaMM lookup failed for {city_name} (Ambient: {avg_temp_c:.1f}C). Cannot proceed.")


        soh_history_df = forecast_fleet_degradation(
            years=forecast_years, initial_soh_cap=initial_soh_cap, initial_soh_res_percent=initial_soh_res,
            cycles_per_day=cycles_per_day, avg_temp_c=rep_avg_temp, peak_temp_c=rep_peak_temp,
            traffic_level=traffic_level, user_profile=user_profile, chemistry=chemistry
        )
        if soh_history_df.empty or soh_history_df['Avg_SoH_Cap_Percent'].isnull().any() or soh_history_df['Resistance_Percent'].isnull().any():
            raise ValueError("Degradation forecast produced invalid results (empty or NaN).")
        final_soh_cap = soh_history_df['Avg_SoH_Cap_Percent'].iloc[-1]; final_res_pct = soh_history_df['Resistance_Percent'].iloc[-1]


        if 'city_renewable_df_global' not in globals() or city_renewable_df_global is None:
             raise RuntimeError("Global city renewable dataframe not loaded.")
        renewable_supply_df = load_renewable_supply_from_city_data(
            city_renewable_df=city_renewable_df_global, current_city_name=city_name,
            city_match_col=renew_city_col, power_val_col=renew_power_col,
            start_date=sim_start_date, end_date=sim_end_date, initial_capacity_mw=initial_renewable_mw,
            growth_rate=renewable_growth_rate, forecast_years=forecast_years
        )


        results_df = simulate_energy_balance_and_charging(
            soh_forecast_df=soh_history_df, fleet_size_initial=initial_fleet_size,
            growth_rate=fleet_growth_rate, daily_km=avg_daily_km, efficiency=base_efficiency,
            nominal_capacity=batt_nom_cap, supply_df=renewable_supply_df, charger_mix=local_charger_mix,
            resistance_impact_factor=resistance_impact, smart_charging_strategy=smart_strategy,
            charge_start_hour=charge_start, charge_end_hour=charge_end,

            initial_max_grid_power_kw=initial_max_grid_power_kw_cfg,
            grid_limit_growth_rate=grid_limit_growth_rate_cfg,
            use_grid_storage=use_storage, storage_capacity_kwh=storage_cap,
            storage_max_charge_kw=storage_chg_kw, storage_max_discharge_kw=storage_dis_kw,
            storage_efficiency=storage_eff, forecast_years=forecast_years
        )


        print("\nCalculating Summary Metrics...")
        total_charged_kwh = results_df['energy_charged_this_hour_kwh'].sum()
        total_unmet_kwh = results_df['unmet_demand_kwh'].sum()
        approx_total_need_kwh = total_charged_kwh + total_unmet_kwh
        approx_total_need_gwh = approx_total_need_kwh / 1e6
        total_charged_gwh = total_charged_kwh / 1e6; total_unmet_gwh = total_unmet_kwh / 1e6
        total_ren_direct_gwh = results_df['demand_met_by_renewables_kwh'].sum() / 1e6
        total_stor_discharge_gwh = results_df['demand_met_by_storage_kwh'].sum() / 1e6
        total_grid_gwh = results_df['demand_met_by_grid_kwh'].sum() / 1e6
        peak_grid_mw = results_df['grid_draw_kw'].max() / 1e3 if 'grid_draw_kw' in results_df.columns and not results_df['grid_draw_kw'].empty else 0
        if total_charged_kwh > 1e-6:
             percent_direct_ren = (total_ren_direct_gwh / total_charged_gwh) * 100
             percent_storage_contrib = (total_stor_discharge_gwh / total_charged_gwh) * 100
             percent_grid_reliance = (total_grid_gwh / total_charged_gwh) * 100
        else: percent_direct_ren=0; percent_storage_contrib=0; percent_grid_reliance=0
        if approx_total_need_kwh > 1e-6: percent_unmet_need = (total_unmet_gwh / approx_total_need_gwh) * 100
        else: percent_unmet_need = 0


        total_ren_used_for_storage = results_df['charge_to_storage_from_renewables_kwh'].sum() / 1e6


        ren_via_storage_gwh = total_ren_used_for_storage * storage_eff

        total_ren_supply_gwh = (results_df['supply_kw'] * 1.0).sum() / 1e6
        percent_ren_utilized = 0
        if total_ren_supply_gwh > 1e-6:
            total_ren_used_gwh = total_ren_direct_gwh + total_ren_used_for_storage
            percent_ren_utilized = (total_ren_used_gwh / total_ren_supply_gwh) * 100

        summary = {
            "City": city_name, "Avg_Ambient_Temp_C": avg_temp_c, "Population": population,
            "Sim_Avg_Batt_Temp_C": rep_avg_temp, "Sim_Peak_Batt_Temp_C": rep_peak_temp,
            "Final_Fleet_Avg_SoH_Cap_%": final_soh_cap, "Final_Fleet_Avg_Resistance_%": final_res_pct,
            "Total_Simulated_EV_Need_GWh": approx_total_need_gwh, "Total_Energy_Charged_GWh": total_charged_gwh,
            "Total_Unmet_Demand_GWh": total_unmet_gwh, "Unmet_Need_Percent": percent_unmet_need,
            "Peak_Grid_Load_MW": peak_grid_mw, "Energy_from_Direct_Renewables_GWh": total_ren_direct_gwh,
            "Energy_from_Storage_GWh": total_stor_discharge_gwh,
            "Energy_from_Grid_GWh": total_grid_gwh,
            "Direct_Renewable_Charge_%": percent_direct_ren,
            "Storage_Charge_Contribution_%": percent_storage_contrib,

            "Grid_Charge_Reliance_%": percent_grid_reliance,
            "Total_Renewable_Energy_Used_GWh": total_ren_direct_gwh + total_ren_used_for_storage,
            "Renewable_Energy_Utilization_%": percent_ren_utilized
        }


        sim_elapsed = timer.time() - sim_start_time
        print(f"------ Full Simulation for {city_name} completed in {sim_elapsed:.2f} seconds ------")


        return summary, soh_history_df, results_df

    except Exception as sim_error:
        print(f"\n!!!!!! ERROR during simulation run for '{city_name}' !!!!!!")
        traceback.print_exc()

        raise sim_error



def plot_soh_forecast(soh_df, city_name="", ax=None):
    """Plots SoH forecast on a given Matplotlib Axes object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if soh_df.empty:
        print(f"WARN: No SoH data to plot for {city_name}.")
        ax.text(0.5, 0.5, "No SoH Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Forecasted Fleet Average State of Health ({city_name})' if city_name else 'Forecasted Fleet Average State of Health')
        return

    title = f'{city_name}' if city_name else 'Forecasted Fleet SoH'
    c1 = 'tab:red'; ax.set_xlabel('Forecast Year'); ax.set_ylabel('Capacity SoH (%)', color=c1)
    ax.plot(soh_df['Year'], soh_df['Avg_SoH_Cap_Percent'], color=c1, marker='o', linestyle='-', linewidth=2, label='Capacity SoH')
    ax.tick_params(axis='y', labelcolor=c1); ax.set_ylim(bottom=max(0, soh_df['Avg_SoH_Cap_Percent'].min() - 10), top=105); ax.grid(True, axis='y', ls=':', alpha=0.7)

    ax2 = ax.twinx(); c2 = 'tab:blue'; ax2.set_ylabel('Resistance (% of Initial)', color=c2)
    if 'Resistance_Percent' in soh_df.columns:
         ax2.plot(soh_df['Year'], soh_df['Resistance_Percent'], color=c2, marker='s', ls='--', linewidth=2, label='Resistance %'); ax2.tick_params(axis='y', labelcolor=c2)
         min_r = soh_df['Resistance_Percent'].min(); max_r = soh_df['Resistance_Percent'].max()
         ax2.set_ylim(bottom=max(95, min_r * 0.98), top=max(110, max_r * 1.05)); ax2.grid(True, axis='y', ls=':', alpha=0.5)
    else: print(f"WARN: No Resistance % data available for plotting {city_name}.")

    l1, b1 = ax.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels();

    ax.legend(l1 + l2, b1 + b2, loc='best', fontsize='small')
    ax.set_title(title)


def plot_energy_balance_sample(energy_df, city_name="", sample_days=7, use_storage=False, ax=None):
    """Plots hourly energy balance sample on a given Matplotlib Axes object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))
    else:
        fig = ax.figure

    title = f'{city_name}' if city_name else f'Hourly Energy Balance (First {sample_days} Days)'

    if energy_df.empty:
        print(f"WARN: No energy data to plot for {city_name}.")
        ax.text(0.5, 0.5, "No Energy Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return
    if len(energy_df) < 24:
        print(f"WARN: Less than 24 hours of energy data available for {city_name}. Cannot plot daily sample.")
        ax.text(0.5, 0.5, "Insufficient Data (<24h)", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return

    days = min(sample_days, len(energy_df) // 24);
    if days == 0:
        print(f"WARN: Less than 24 hours of data after division for {city_name}. Cannot plot daily sample.");
        ax.text(0.5, 0.5, "Insufficient Data (<1 day)", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return
    if days < sample_days: title = f'{city_name}'

    sample_data = energy_df.head(24 * days).copy()
    plot_cols = ['demand_met_by_renewables_kwh', 'demand_met_by_storage_kwh', 'demand_met_by_grid_kwh', 'unmet_demand_kwh', 'supply_kw', 'total_hourly_potential_charge_need_kw', 'storage_soc_kwh']
    for col in plot_cols:
        if col not in sample_data.columns: sample_data[col] = 0.0

    sample_data['total_demand_kwh'] = (sample_data['demand_met_by_renewables_kwh'] + sample_data.get('demand_met_by_storage_kwh', 0) + sample_data.get('demand_met_by_grid_kwh', 0) + sample_data.get('unmet_demand_kwh', 0))
    stack_components = {'demand_met_by_renewables_kwh': 'Met by Renewables', 'demand_met_by_storage_kwh': 'Met by Storage', 'demand_met_by_grid_kwh': 'Met by Grid', 'unmet_demand_kwh': 'Unmet Demand'}
    active_components = {k: v for k, v in stack_components.items() if k in sample_data.columns and sample_data[k].sum() > 1e-6}
    colors = {'Met by Renewables': '#90ee90', 'Met by Storage': '#add8e6', 'Met by Grid': '#ffcccb', 'Unmet Demand': '#a9a9a9'}

    if active_components:
        ax.stackplot(sample_data.index, sample_data[active_components.keys()].values.T, labels=active_components.values(), alpha=0.8, colors=[colors.get(label, '#cccccc') for label in active_components.values()])

    ax.plot(sample_data.index, sample_data['supply_kw'], label='Renewable Supply (kW)', color='darkgreen', linestyle='-', linewidth=1.5, alpha=0.8)
    if 'total_hourly_potential_charge_need_kw' in sample_data.columns:
        ax.plot(sample_data.index, sample_data['total_hourly_potential_charge_need_kw'], label='Potential EV Charge Need (kW)', color='black', linestyle=':', linewidth=1.5)

    ax.set_xlabel('Time'); ax.set_ylabel('Power (kW) / Energy (kWh)');
    ax.grid(True, axis='both', ls=':', alpha=0.6); ax.set_ylim(bottom=0);
    ax.tick_params(axis='x', rotation=15)

    handles, labels = ax.get_legend_handles_labels()

    if use_storage and 'storage_soc_kwh' in sample_data.columns and sample_data['storage_soc_kwh'].max() > 1e-6:
        ax_storage = ax.twinx(); color_storage = 'purple';
        ax_storage.set_ylabel('Grid Storage SoC (kWh)', color=color_storage)
        line_storage, = ax_storage.plot(sample_data.index, sample_data['storage_soc_kwh'], color=color_storage, linestyle='-.', linewidth=2, label='Storage SoC')
        ax_storage.tick_params(axis='y', labelcolor=color_storage)
        max_soc = sample_data['storage_soc_kwh'].max();
        ax_storage.set_ylim(bottom=-max_soc * 0.05, top=max_soc * 1.05 if max_soc > 0 else 1)

        handles.append(line_storage); labels.append('Storage SoC')

        ax_storage.spines['right'].set_position(('outward', 5))


    ax.legend(handles, labels, loc='upper left', fontsize='small')
    ax.set_title(title)


def plot_yearly_peaks(energy_df, city_name="", ax=None):
    """Plots yearly peak grid draw on a given Matplotlib Axes object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    title = f'{city_name}' if city_name else 'Yearly Peak Grid Draw'

    if energy_df.empty or 'grid_draw_kw' not in energy_df.columns:
        print(f"WARN: No peak grid draw data available to plot for {city_name}.")
        ax.text(0.5, 0.5, "No Peak Grid Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return


    df_copy = energy_df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        try: df_copy.index = pd.to_datetime(df_copy.index)
        except Exception as e:
            print(f"ERROR: Could not convert index for peak plot ({city_name}): {e}")
            ax.text(0.5, 0.5, "Index Error", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); return

    try:
        if df_copy.index.tz is not None:
            df_copy.index = df_copy.index.tz_convert(None)


        daily_max = df_copy['grid_draw_kw'].resample('D').max()
        yearly_peak = daily_max.resample('YE').max() / 1000

    except Exception as e:
        print(f"ERROR: Failed resampling for yearly peaks plot ({city_name}): {e}"); traceback.print_exc()
        ax.text(0.5, 0.5, "Resampling Error", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return

    if yearly_peak.empty or yearly_peak.isnull().all():
        print(f"WARN: No valid peak grid draw data found after resampling for {city_name}.")
        ax.text(0.5, 0.5, "No Valid Peak Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return

    yearly_peak.index = yearly_peak.index.year
    yearly_peak.plot(kind='bar', color='firebrick', edgecolor='black', ax=ax)
    ax.set_title(title); ax.set_xlabel('Year'); ax.set_ylabel('Peak Grid Draw (MW)');
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', linestyle=':')


def plot_yearly_energy_contribution(energy_df, city_name="", ax=None):
    """Plots yearly energy source contribution on a given Matplotlib Axes object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    title = f'{city_name}' if city_name else 'Yearly Energy Source Contribution'

    if energy_df.empty:
        print(f"WARN: No energy data for {city_name}.")
        ax.text(0.5, 0.5, "No Energy Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return


    df_copy = energy_df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        try: df_copy.index = pd.to_datetime(df_copy.index)
        except Exception:
             print(f"ERROR: Cannot convert index for energy contribution plot ({city_name}).")
             ax.text(0.5, 0.5, "Index Error", ha='center', va='center', transform=ax.transAxes)
             ax.set_title(title); return

    try:
        if df_copy.index.tz is not None:
            df_copy.index = df_copy.index.tz_convert(None)

        cols_to_sum = ['demand_met_by_renewables_kwh', 'demand_met_by_storage_kwh', 'demand_met_by_grid_kwh', 'unmet_demand_kwh']
        available_cols = [col for col in cols_to_sum if col in df_copy.columns]
        if not available_cols:
            print(f"WARN: No relevant energy columns found for energy contribution ({city_name}).")
            ax.text(0.5, 0.5, "Missing Data Columns", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); return

        yearly_data = df_copy[available_cols].resample('YE').sum() / 1e6
        yearly_data = yearly_data.loc[:, yearly_data.sum() > 1e-6]
        if yearly_data.empty:
            print(f"WARN: No non-zero yearly energy contribution data to plot for {city_name}.")
            ax.text(0.5, 0.5, "No Contribution Data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); return

        col_rename_map = {'demand_met_by_renewables_kwh': 'Renewables', 'demand_met_by_storage_kwh': 'Storage', 'demand_met_by_grid_kwh': 'Grid', 'unmet_demand_kwh': 'Unmet Demand'}
        yearly_data.columns = [col_rename_map.get(col, col) for col in yearly_data.columns]
        yearly_data.index = yearly_data.index.year

        plot_colors = {'Renewables':'#90ee90', 'Storage':'#add8e6', 'Grid':'#ffcccb', 'Unmet Demand':'#a9a9a9'}
        yearly_data.plot(kind='bar', stacked=True, ax=ax, color=[plot_colors.get(col, '#cccccc') for col in yearly_data.columns])
        ax.set_title(title); ax.set_xlabel('Year'); ax.set_ylabel('Energy (GWh)');
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title="Energy Source", fontsize='small'); ax.grid(True, axis='y', linestyle=':');

    except Exception as e:
        print(f"ERROR generating yearly energy contribution plot ({city_name}): {e}"); traceback.print_exc()
        ax.text(0.5, 0.5, "Plotting Error", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)


def plot_yearly_renewable_utilization(energy_df, city_name="", ax=None):
    """Plots yearly renewable energy utilization on a given Matplotlib Axes object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    title = f'{city_name}' if city_name else 'Yearly Renewable Energy Utilization'

    if energy_df.empty:
        print(f"WARN: No energy data for {city_name}.")
        ax.text(0.5, 0.5, "No Energy Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return


    df_copy = energy_df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        try: df_copy.index = pd.to_datetime(df_copy.index)
        except Exception:
            print(f"ERROR: Cannot convert index for renewable utilization plot ({city_name}).")
            ax.text(0.5, 0.5, "Index Error", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); return

    try:
        if df_copy.index.tz is not None:
            df_copy.index = df_copy.index.tz_convert(None)

        required_cols = ['supply_kw', 'unused_renewables_kw', 'demand_met_by_renewables_kwh', 'charge_to_storage_from_renewables_kwh']
        if not all(col in df_copy.columns for col in required_cols):
            print(f"WARN: Missing columns for renewable utilization ({city_name}). Need: {required_cols}")
            ax.text(0.5, 0.5, "Missing Data Columns", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); return


        hourly_supply_kwh = df_copy['supply_kw'] * 1.0
        yearly_supply_gwh = hourly_supply_kwh.resample('YE').sum() / 1e6
        yearly_unused_gwh = (df_copy['unused_renewables_kw'] * 1.0).resample('YE').sum() / 1e6
        yearly_used_ev_gwh = df_copy['demand_met_by_renewables_kwh'].resample('YE').sum() / 1e6
        yearly_used_storage_gwh = df_copy['charge_to_storage_from_renewables_kwh'].resample('YE').sum() / 1e6

        plot_data = pd.DataFrame({'Used by EVs': yearly_used_ev_gwh, 'Used by Storage': yearly_used_storage_gwh, 'Unused (Curtailed)': yearly_unused_gwh})
        plot_data.index = plot_data.index.year
        if plot_data.sum().sum() < 1e-6 and yearly_supply_gwh.sum() < 1e-6:
             print(f"WARN: No significant renewable utilization data to plot for {city_name}.")
             ax.text(0.5, 0.5, "No Renewable Data", ha='center', va='center', transform=ax.transAxes)
             ax.set_title(title); return


        yearly_supply_gwh.index = yearly_supply_gwh.index.year

        yearly_supply_gwh = yearly_supply_gwh.reindex(plot_data.index).fillna(0)


        plot_data.plot(kind='bar', stacked=True, ax=ax, color={'Used by EVs':'#90ee90', 'Used by Storage':'#3cb371', 'Unused (Curtailed)':'#d3d3d3'})
        x_ticks_loc = np.arange(len(plot_data.index))
        ax.plot(x_ticks_loc, yearly_supply_gwh.values, marker='o', linestyle='-', color='darkblue', label='Total Renewable Supply')
        ax.set_title(title); ax.set_xlabel('Year'); ax.set_ylabel('Energy (GWh)');
        ax.set_xticks(ticks=x_ticks_loc)
        ax.set_xticklabels(labels=plot_data.index, rotation=45, ha='right');
        ax.legend(title="Renewable Energy Fate", fontsize='small'); ax.grid(True, axis='y', linestyle=':');

    except Exception as e:
        print(f"ERROR generating yearly renewable utilization plot ({city_name}): {e}"); traceback.print_exc()
        ax.text(0.5, 0.5, "Plotting Error", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)


def plot_average_daily_charging_profile(energy_df, city_name="", ax=None):
    """Plots average daily potential EV charging profile on a given Matplotlib Axes object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    title = f'{city_name}' if city_name else 'Average Daily Potential EV Charging Profile'

    if energy_df.empty or 'total_hourly_potential_charge_need_kw' not in energy_df.columns:
        print(f"WARN: No potential charge need data for {city_name}.")
        ax.text(0.5, 0.5, "No Charging Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return

    try:

        df_copy = energy_df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index)

        hourly_avg_demand = df_copy.groupby(df_copy.index.hour)['total_hourly_potential_charge_need_kw'].mean()
        if hourly_avg_demand.empty or hourly_avg_demand.isnull().all():
            print(f"WARN: No valid average hourly demand data to plot for {city_name}.")
            ax.text(0.5, 0.5, "No Average Data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); return

        hourly_avg_demand.plot(kind='line', marker='.', ax=ax, color='teal')
        ax.set_title(title); ax.set_xlabel('Hour of Day (0-23)'); ax.set_ylabel('Average Potential Demand (kW)');
        ax.set_xticks(np.arange(0, 24, 2)); ax.grid(True, axis='both', linestyle=':');
        ax.set_ylim(bottom=min(0, hourly_avg_demand.min() - hourly_avg_demand.max()*0.05) if hourly_avg_demand.max()>0 else 0)

    except Exception as e:
        print(f"ERROR generating average daily charging profile plot ({city_name}): {e}"); traceback.print_exc()
        ax.text(0.5, 0.5, "Plotting Error", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)

def plot_yearly_unmet_demand(energy_df, city_name="", ax=None):
    """Plots total yearly unmet EV charging demand on a given Matplotlib Axes object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    title = f'{city_name}' if city_name else 'Total Yearly Unmet EV Charging Demand'

    if energy_df.empty or 'unmet_demand_kwh' not in energy_df.columns:
        print(f"WARN: No unmet demand data for {city_name}.")
        ax.text(0.5, 0.5, "No Unmet Data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title); return


    df_copy = energy_df.copy()
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        try: df_copy.index = pd.to_datetime(df_copy.index)
        except Exception:
            print(f"ERROR: Cannot convert index for unmet demand plot ({city_name}).")
            ax.text(0.5, 0.5, "Index Error", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title); return

    try:
        if df_copy.index.tz is not None:
            df_copy.index = df_copy.index.tz_convert(None)

        yearly_unmet_gwh = df_copy['unmet_demand_kwh'].resample('YE').sum() / 1e6


        yearly_unmet_gwh.index = yearly_unmet_gwh.index.year
        yearly_unmet_gwh.plot(kind='bar', ax=ax, color='darkred', edgecolor='black')
        ax.set_title(title); ax.set_xlabel('Year'); ax.set_ylabel('Unmet Demand (GWh)');
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', linestyle=':')
        min_val = yearly_unmet_gwh.min(); max_val = yearly_unmet_gwh.max()
        ax.set_ylim(bottom=min(0, min_val - max_val*0.05 if max_val > 0 else 0), top=max(1, max_val * 1.1 if max_val > 0 else 1))

    except Exception as e:
        print(f"ERROR generating yearly unmet demand plot ({city_name}): {e}"); traceback.print_exc()
        ax.text(0.5, 0.5, "Plotting Error", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)


cities_df_global = None
city_renewable_capacity_df_global = None
renewable_profiles_data_df_global = None

if __name__ == "__main__":
    print("--- Starting TWO-CITY Comparison EV Degradation & Energy Simulation ---")
    print("--- >> Using Pre-computation and Caching for Speed << ---")
    overall_start_time = timer.time()


    if not CITIES_CSV_PATH or not CITIES_CSV_PATH.is_file(): print(f"ERROR: City list file not found: '{CITIES_CSV_PATH}'."); sys.exit()
    if not CITY_RENEWABLE_DATA_PATH or not CITY_RENEWABLE_DATA_PATH.is_file(): print(f"ERROR: City renewable data file not found: '{CITY_RENEWABLE_DATA_PATH}'."); sys.exit()

    try: load_pybamm_lookup()
    except (FileNotFoundError, RuntimeError) as e: sys.exit(f"Exiting: {e}")


    try:
        print(f"\nLoading city list data from: {CITIES_CSV_PATH}")
        cities_df_global = pd.read_csv(CITIES_CSV_PATH)
        required_cols = [CITY_COL, TEMP_COL, POP_COL]; assert all(col in cities_df_global.columns for col in required_cols), f"Cities CSV missing cols: {required_cols}"
        cities_df_global[TEMP_COL] = pd.to_numeric(cities_df_global[TEMP_COL], errors='coerce')
        cities_df_global[POP_COL] = pd.to_numeric(cities_df_global[POP_COL], errors='coerce')
        cities_df_global.dropna(subset=[CITY_COL, TEMP_COL, POP_COL], inplace=True)
        cities_df_global = cities_df_global[cities_df_global[POP_COL] > 0].reset_index(drop=True)
        cities_df_global[POP_COL] = cities_df_global[POP_COL].astype(int)
        cities_df_global[CITY_COL] = cities_df_global[CITY_COL].astype(str).str.strip()
        print(f"Loaded data for {len(cities_df_global)} valid cities.")
        if len(cities_df_global) == 0: raise SystemExit("Exiting: No valid cities found.")

        print(f"\nLoading city renewable data from: {CITY_RENEWABLE_DATA_PATH}")
        city_renewable_df_global = pd.read_csv(CITY_RENEWABLE_DATA_PATH)
        required_renew_cols = [CITY_RENEW_CITY_COL, CITY_RENEW_POWER_COL]; assert all(col in city_renewable_df_global.columns for col in required_renew_cols), f"Renewable CSV missing cols: {required_renew_cols}"
        city_renewable_df_global[CITY_RENEW_POWER_COL] = pd.to_numeric(city_renewable_df_global[CITY_RENEW_POWER_COL], errors='coerce')
        city_renewable_df_global.dropna(subset=[CITY_RENEW_CITY_COL, CITY_RENEW_POWER_COL], inplace=True)
        city_renewable_df_global[CITY_RENEW_CITY_COL] = city_renewable_df_global[CITY_RENEW_CITY_COL].astype(str).str.strip()
        duplicates = city_renewable_df_global[city_renewable_df_global.duplicated(subset=[CITY_RENEW_CITY_COL], keep=False)]
        if not duplicates.empty: print(f"WARN: Duplicate cities in renewable data: {duplicates[CITY_RENEW_CITY_COL].unique().tolist()}. Keeping first.")
        city_renewable_df_global = city_renewable_df_global.drop_duplicates(subset=[CITY_RENEW_CITY_COL], keep='first')
        print(f"Loaded renewable data for {len(city_renewable_df_global)} unique cities.")

    except Exception as e: print(f"ERROR loading data files: {e}"); traceback.print_exc(); sys.exit()



    print("\n--- City Selection for Comparison ---")
    display_limit = min(20, len(cities_df_global)); print("Available cities (sample):"); print(cities_df_global[CITY_COL].head(display_limit).to_string(index=False))
    if len(cities_df_global) > display_limit: print("...")

    selected_cities = []
    city_data_rows = {}

    for i in range(1, 3):
        target_city_name = ""
        while not target_city_name:
            try: target_city_name = input(f"Enter exact name for City {i} (case-sensitive): ").strip()
            except EOFError: print("\nInput stream closed."); sys.exit()
            if not target_city_name: print(f"City {i} name cannot be empty.")
            elif i == 2 and target_city_name == selected_cities[0]:
                 print("Please select two different cities.")
                 target_city_name = ""


        target_city_row = cities_df_global[cities_df_global[CITY_COL] == target_city_name]

        if target_city_row.empty:
            print(f"\nERROR: City '{target_city_name}' not found in list '{CITIES_CSV_PATH}'. Check spelling/case.")
            if HAS_FUZZYWUZZY:
                try:
                    suggestions = fuzzy_process.extract(target_city_name, cities_df_global[CITY_COL].tolist(), limit=5)
                    print("\nDid you mean one of these?")
                    for suggestion, score in suggestions:
                        if score > 70: print(f" - {suggestion}")
                except Exception: pass
            print(f"Exiting: Target city {i} not found.")
            sys.exit()
        else:
            city_row = target_city_row.iloc[0]
            city_name_selected = city_row[CITY_COL]; avg_temp_selected = city_row[TEMP_COL]; population_selected = city_row[POP_COL]
            print(f"Selected City {i}: {city_name_selected} (Pop: {population_selected:,.0f}, Temp: {avg_temp_selected:.1f}°C)")
            selected_cities.append(city_name_selected)
            city_data_rows[city_name_selected] = city_row


    charger_mix_items_sorted = sorted(CHARGER_MIX.items())
    charger_mix_tuple_of_tuples = []
    for charger_name, details_dict in charger_mix_items_sorted:
        inner_tuple = tuple(sorted(details_dict.items()))
        charger_mix_tuple_of_tuples.append((charger_name, inner_tuple))
    charger_mix_tuple_for_cache = tuple(charger_mix_tuple_of_tuples)

    current_config_tuple = (
        FORECAST_YEARS, FLEET_GROWTH_RATE, RENEWABLE_GROWTH_RATE, AVG_DAILY_KM_PER_EV,
        BASE_EFFICIENCY_KWH_KM, CYCLES_PER_DAY, CHEMISTRY, TRAFFIC_LEVEL, USER_PROFILE,
        PYBAMM_PARAM_SET, PYBAMM_SIM_TYPE, PYBAMM_SIM_CRATE, HEAT_TRANSFER_COEFF, PYBAMM_INITIAL_SOC,
        charger_mix_tuple_for_cache,
        RESISTANCE_IMPACT_FACTOR, CHARGE_START_HOUR, CHARGE_END_HOUR, SMART_CHARGING_STRATEGY,
        USE_GRID_STORAGE, STORAGE_CAPACITY_KWH, STORAGE_MAX_CHARGE_KW, STORAGE_MAX_DISCHARGE_KW, STORAGE_EFFICIENCY,
        INITIAL_AVG_SOH_PERCENT, INITIAL_AVG_RESISTANCE_PERCENT, BATTERY_NOMINAL_CAPACITY_KWH, CITY_RENEW_POWER_COL,
        CITY_RENEW_CITY_COL,
        INITIAL_MAX_GRID_POWER_KW, GRID_LIMIT_GROWTH_RATE
    )


    city_results = {}
    simulation_successful = True
    for city_name in selected_cities:
        city_row = city_data_rows[city_name]
        avg_temp_selected = city_row[TEMP_COL]
        population_selected = city_row[POP_COL]

        print(f"\n===== Running Simulation for: {city_name} =====")
        try:
            call_start_time = timer.time()
            summary, soh_df, results_df = run_full_city_simulation(
                city_name=city_name,
                avg_temp_c=avg_temp_selected,
                population=population_selected,
                config_tuple=current_config_tuple
            )
            call_end_time = timer.time()
            print(f"--- Simulation function call for {city_name} returned in {call_end_time - call_start_time:.2f} sec ---")
            city_results[city_name] = {"summary": summary, "soh": soh_df, "energy": results_df}

        except Exception as main_error:
            print(f"\n!!!!!! SIMULATION FAILED for {city_name} !!!!!!")
            traceback.print_exc()
            city_results[city_name] = None
            simulation_successful = False


    print("\n\n----- Simulation Results Summaries -----")
    for city_name in selected_cities:
        print(f"\n--- Summary for {city_name} ---")
        if city_results[city_name]:
            summary = city_results[city_name]["summary"]
            for key, value in summary.items():
                if isinstance(value, float): print(f"  {key}: {value:,.2f}")
                elif isinstance(value, int): print(f"  {key}: {value:,d}")
                else: print(f"  {key}: {value}")


            try:
                summary_df_single = pd.DataFrame([summary])
                safe_city_name = "".join(c if c.isalnum() else "_" for c in city_name)
                single_city_output_path = Path(f"./{safe_city_name}_{SUMMARY_OUTPUT_FILENAME_BASE}.csv")
                summary_df_single.to_csv(single_city_output_path, index=False, float_format='%.3f')
                print(f"\nSummary saved to: {single_city_output_path.resolve()}")
            except Exception as e: print(f"\nWarning: Could not save summary CSV for {city_name}: {e}")
        else:
            print("  ERROR: Simulation failed for this city, no summary available.")


    if simulation_successful and len(city_results) == 2:
        print("\nGenerating comparison plots...")
        city1_name = selected_cities[0]
        city2_name = selected_cities[1]
        city1_data = city_results[city1_name]
        city2_data = city_results[city2_name]


        fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        plot_soh_forecast(city1_data["soh"], city_name=city1_name, ax=axes1[0])
        plot_soh_forecast(city2_data["soh"], city_name=city2_name, ax=axes1[1])
        fig1.suptitle("Comparison: Forecasted Fleet Average State of Health", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])


        fig2, axes2 = plt.subplots(1, 2, figsize=(20, 7), sharey=True)
        plot_energy_balance_sample(city1_data["energy"], city_name=city1_name, sample_days=14, use_storage=USE_GRID_STORAGE, ax=axes2[0])
        plot_energy_balance_sample(city2_data["energy"], city_name=city2_name, sample_days=14, use_storage=USE_GRID_STORAGE, ax=axes2[1])

        handles, labels = axes2[0].get_legend_handles_labels()
        fig2.legend(handles, labels, loc='upper center', ncol=len(labels)//2 + len(labels)%2, bbox_to_anchor=(0.5, 0.98), fontsize='small')
        axes2[0].legend_ = None
        axes2[1].legend_ = None
        fig2.suptitle("Comparison: Hourly Energy Balance (First 14 Days)", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])


        fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        plot_yearly_peaks(city1_data["energy"], city_name=city1_name, ax=axes3[0])
        plot_yearly_peaks(city2_data["energy"], city_name=city2_name, ax=axes3[1])
        fig3.suptitle("Comparison: Yearly Peak Grid Draw", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])


        fig4, axes4 = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        plot_yearly_energy_contribution(city1_data["energy"], city_name=city1_name, ax=axes4[0])
        plot_yearly_energy_contribution(city2_data["energy"], city_name=city2_name, ax=axes4[1])

        handles, labels = axes4[0].get_legend_handles_labels()
        fig4.legend(handles, labels, loc='upper center', ncol=len(labels), bbox_to_anchor=(0.5, 0.98), title="Energy Source", fontsize='small')
        axes4[0].legend_ = None
        axes4[1].legend_ = None
        fig4.suptitle("Comparison: Yearly Energy Source Contribution", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])


        fig5, axes5 = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        plot_yearly_renewable_utilization(city1_data["energy"], city_name=city1_name, ax=axes5[0])
        plot_yearly_renewable_utilization(city2_data["energy"], city_name=city2_name, ax=axes5[1])

        handles, labels = axes5[0].get_legend_handles_labels()
        fig5.legend(handles, labels, loc='upper center', ncol=len(labels)//2 + len(labels)%2, bbox_to_anchor=(0.5, 0.98), title="Renewable Energy Fate", fontsize='small')
        axes5[0].legend_ = None
        axes5[1].legend_ = None
        fig5.suptitle("Comparison: Yearly Renewable Energy Utilization", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.90])


        fig6, axes6 = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        plot_average_daily_charging_profile(city1_data["energy"], city_name=city1_name, ax=axes6[0])
        plot_average_daily_charging_profile(city2_data["energy"], city_name=city2_name, ax=axes6[1])
        fig6.suptitle("Comparison: Average Daily Potential EV Charging Profile", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])


        fig7, axes7 = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
        plot_yearly_unmet_demand(city1_data["energy"], city_name=city1_name, ax=axes7[0])
        plot_yearly_unmet_demand(city2_data["energy"], city_name=city2_name, ax=axes7[1])
        fig7.suptitle("Comparison: Total Yearly Unmet EV Charging Demand", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])

        print("\nDisplaying comparison plots. Close plot windows to exit.")
        plt.show(block=True)

    elif not simulation_successful:
        print("\nSkipping plot generation as one or both simulations failed.")
    else:
        print("\nUnexpected state: Simulation marked successful but results dictionary is incomplete.")


    overall_elapsed_time = timer.time() - overall_start_time
    cache_info = run_full_city_simulation.cache_info()
    print(f"\n--- Comparison Simulation Complete ({overall_elapsed_time:.2f} seconds) ---")
    print(f"--- Cache Info: Hits={cache_info.hits}, Misses={cache_info.misses}, MaxSize={cache_info.maxsize}, CurrentSize={cache_info.currsize} ---")
