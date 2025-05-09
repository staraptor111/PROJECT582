

import pandas as pd
import pvlib
import os
import pytz
import traceback
import numpy as np

from pvlib.pvsystem import PVSystem, retrieve_sam
from pvlib.location import Location
from pvlib.modelchain import ModelChain




DEFAULT_PSM3_FILE = 'psm3_data_60min.csv'


def find_weather_file(file_path_input):
    """
    Tries to find the weather file based on input, checking multiple locations.

    Args:
        file_path_input (str): The initial path/filename provided.

    Returns:
        str or None: The absolute path to the found file, or None if not found.
    """
    if not file_path_input:
        print("Error: No weather file path provided.")
        return None

    if os.path.isabs(file_path_input):
        if os.path.exists(file_path_input) and os.path.isfile(file_path_input):
             print(f"Found weather file at absolute path: {file_path_input}")
             return file_path_input
        else:
             print(f"Error: Provided absolute path not found or not a file: {file_path_input}")
             return None

    script_dir = os.path.dirname(__file__) if '__file__' in locals() else os.getcwd()
    base_filename = os.path.basename(file_path_input)
    paths_to_check = [
        file_path_input,
        os.path.join(script_dir, file_path_input),
        os.path.join(script_dir, base_filename),
        os.path.join(script_dir, 'data', base_filename),
        os.path.join(script_dir, 'weather_data', base_filename)
    ]
    seen_paths = set()
    unique_paths = []
    for p in paths_to_check:
        abs_p = os.path.abspath(p)
        if abs_p not in seen_paths: seen_paths.add(abs_p); unique_paths.append(p)

    print(f"\nSearching for weather file '{base_filename}' relative to {script_dir}:")
    for p in unique_paths:
        abs_path_check = os.path.abspath(p)
        if os.path.exists(abs_path_check) and os.path.isfile(abs_path_check):
            print(f"Found weather file at: {abs_path_check}")
            return abs_path_check

    print(f"Error: Weather file '{base_filename}' not found relative to script or in common subdirectories.")
    return None


def load_pv_generation(weather_file, location_name="Default Location", tz_str="UTC"):
    """Loads PV generation data from a weather file using pvlib."""
    resolved_path = find_weather_file(weather_file)
    if not resolved_path: return None, None

    try:
        print(f"\nLoading weather data from: {resolved_path}")
        print(f"Using target Timezone: {tz_str}")
        metadata, weather_data, file_type = None, None, "Unknown"

        if str(resolved_path).lower().endswith('.csv'):
            try:
                print("  Attempting to read as PSM3 CSV...")
                weather_data, metadata = pvlib.iotools.read_psm3(resolved_path, map_variables=True)
                file_type = "PSM3"; print("  Successfully read as PSM3.")
            except Exception as e_psm3:
                print(f"  Failed to read as PSM3: {e_psm3}")
                try:
                    print("  Attempting to read as TMY3 CSV...")
                    weather_data, metadata = pvlib.iotools.read_tmy3(resolved_path, map_variables=True)
                    file_type = "TMY3"; print("  Successfully read as TMY3.")
                except Exception as e_tmy3: raise ValueError(f"Could not read file as PSM3 or TMY3: {resolved_path}") from e_tmy3
        else: raise ValueError(f"Unsupported weather file format (expected .csv): {resolved_path}")

        print("\n--- Timezone Processing ---")
        if metadata is None: raise ValueError("Metadata could not be read.")

        file_tz_offset, file_tz_str = None, None
        tz_key, offset_val = ('Time Zone', None) if file_type == "PSM3" else ('TZ', None)
        try:
            offset_val = metadata.get(tz_key, 'NOT_FOUND')
            if offset_val != 'NOT_FOUND': file_tz_offset = int(offset_val)
            if file_tz_offset is not None:
                 file_tz_str = f'Etc/GMT{-file_tz_offset:+d}'
                 print(f"  {file_type} Metadata '{tz_key}' offset found: {file_tz_offset} -> Interpreting as {file_tz_str}")
            else: print(f"  {file_type} Metadata '{tz_key}' key not found or invalid ('{offset_val}').")
        except Exception as e: print(f"  Warning: Error processing {file_type} '{tz_key}' metadata: {e}")

        if weather_data.index.tz is not None:
            print(f"  Weather data index already tz-aware: {weather_data.index.tz}. Checking conversion need.")
            file_tz_str = str(weather_data.index.tz)
        elif file_tz_str:
            try: weather_data.index = weather_data.index.tz_localize(file_tz_str); print(f"  Localized naive index using: {file_tz_str}")
            except Exception as e: print(f"  ERROR localizing with {file_tz_str}: {e}. Falling back."); file_tz_str = None
        if not weather_data.index.tz and not file_tz_str :
            print(f"  Warning: Assuming naive timestamps are in target timezone: {tz_str}")
            try:
                weather_data.index = weather_data.index.tz_localize(tz_str, ambiguous='infer', nonexistent='shift_forward')
                file_tz_str = tz_str; print(f"  Localization using target {tz_str} successful (used infer/shift).")
            except Exception as e: raise ValueError("Failed to localize timestamp index.") from e

        if weather_data.index.tz is None: raise ValueError("Timezone localization failed.")

        target_tz = pytz.timezone(tz_str)
        if weather_data.index.tz.zone != target_tz.zone:
             print(f"  Converting timezone from {weather_data.index.tz.zone} to target {target_tz.zone}...")
             try: weather_data.index = weather_data.index.tz_convert(target_tz); print(f"  Conversion successful. Index TZ: {weather_data.index.tz}")
             except Exception as e: print(f"  ERROR during timezone conversion: {e}"); raise
        else: print(f"  File timezone already matches target: {target_tz.zone}")

        print("\n--- Location Definition ---")
        lat = metadata.get('latitude', metadata.get('Latitude'))
        lon = metadata.get('longitude', metadata.get('Longitude'))
        alt = metadata.get('altitude', metadata.get('Altitude', metadata.get('Elevation')))
        if lat is None or lon is None: raise ValueError(f"Lat/Lon not found in metadata. Keys: {metadata.keys()}")
        if alt is None: print(f"  Warning: Altitude not found. Using 0. Keys: {metadata.keys()}"); alt = 0
        try: lat, lon, alt = float(lat), float(lon), float(alt); print(f"  Using Lat: {lat}, Lon: {lon}, Alt: {alt}")
        except Exception as e: raise ValueError(f"Cannot convert lat/lon/alt to float. Values: {lat}, {lon}, {alt}") from e

        location = Location(latitude=lat, longitude=lon, tz=target_tz.zone, altitude=alt, name=location_name)
        print(f"Created Location object: {location.name} ({location.latitude:.4f}, {location.longitude:.4f}, Alt={location.altitude}, TZ='{location.tz}')")

        print("\nWeather Data Loaded:")
        print("Columns:", list(weather_data.columns)); print(weather_data.head())
        req_cols = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
        if missing:= [c for c in req_cols if c not in weather_data.columns]: print(f"\nWarning: Missing essential columns: {missing}")

        return weather_data, location

    except Exception as e:
        print(f"\n*** ERROR in load_pv_generation: {e} ***")
        if 'metadata' in locals() and metadata: print(f"  Metadata at error: {metadata}")
        traceback.print_exc(); return None, None

def create_load_profile(index, baseload_kw, ev_kw, ev_start, ev_end, ev_freq_days):
    """Creates a simple hourly load profile Series."""
    load = pd.Series(baseload_kw, index=index, name="load_kw")
    current_day = -1
    days_since_charge = ev_freq_days
    for timestamp in index:
        day_of_year, hour = timestamp.dayofyear, timestamp.hour
        if day_of_year != current_day: current_day = day_of_year; days_since_charge += 1
        is_charge_day = (days_since_charge >= ev_freq_days)
        is_charge_hour = (hour >= ev_start or hour < ev_end) if ev_end <= ev_start else (hour >= ev_start and hour < ev_end)
        if is_charge_day and is_charge_hour:
            load[timestamp] += ev_kw

            is_end_hour = (hour == ev_end - 1) if ev_end > 0 else (hour == 23 and ev_end == 0)
            if is_end_hour: days_since_charge = 0
    return load

def simulate_battery(pv_generation_kw, load_kw, params):
    """Simulates battery operation hour by hour."""
    if not pv_generation_kw.index.equals(load_kw.index): raise ValueError("PV/Load indices mismatch!")
    index = pv_generation_kw.index; n_steps = len(index)
    tdiff = index.to_series().diff().median(); hrs = tdiff.total_seconds()/3600 if pd.notna(tdiff) and tdiff.total_seconds()>0 else 1.0

    soc_kwh, batt_ch_kw, batt_dis_kw, unmet_kw, curtail_kw = (np.zeros(n_steps) for _ in range(5))
    current_soc = params['initial_soc_kwh']

    for i in range(n_steps):
        pv = pv_generation_kw.iloc[i]; load = load_kw.iloc[i]
        net = pv - load

        if net >= 0:
            p_to_charge = net
            p_to_charge = min(p_to_charge, params['max_charge_kw'])
            space_kwh = params['capacity_kwh'] - current_soc
            p_lim_by_space = space_kwh / (hrs * params['charge_eff']) if hrs * params['charge_eff'] > 1e-6 else float('inf')
            p_to_charge = min(p_to_charge, p_lim_by_space)
            p_to_charge = max(0, p_to_charge)

            e_charged = p_to_charge * hrs * params['charge_eff']
            current_soc += e_charged

            batt_ch_kw[i] = p_to_charge
            curtail_kw[i] = net - p_to_charge
            unmet_kw[i] = 0; batt_dis_kw[i] = 0
        else:
            deficit = abs(net)
            p_needed_from_batt = deficit / params['discharge_eff'] if params['discharge_eff'] > 1e-6 else float('inf')

            p_to_discharge = min(p_needed_from_batt, params['max_discharge_kw'])
            e_avail = current_soc - params['min_soc_kwh']
            p_lim_by_energy = e_avail / hrs if hrs > 1e-6 else float('inf')
            p_to_discharge = min(p_to_discharge, p_lim_by_energy)
            p_to_discharge = max(0, p_to_discharge)

            e_discharged = p_to_discharge * hrs
            current_soc -= e_discharged
            p_delivered = p_to_discharge * params['discharge_eff']

            batt_dis_kw[i] = p_to_discharge
            unmet = deficit - p_delivered
            unmet_kw[i] = max(0, unmet)
            batt_ch_kw[i] = 0; curtail_kw[i] = 0

        soc_kwh[i] = current_soc

    results = pd.DataFrame({
        'pv_gen_kw': pv_generation_kw, 'load_kw': load_kw, 'soc_kwh': soc_kwh,
        'batt_ch_kw': batt_ch_kw, 'batt_dis_kw': batt_dis_kw,
        'unmet_kw': unmet_kw, 'curtail_kw': curtail_kw }, index=index)
    results['soc_percent'] = (results['soc_kwh'] / params['capacity_kwh']) * 100.0
    return results, hrs


if __name__ == "__main__":
    print("=====================================")
    print("Starting PV + Off-Grid Simulation Script")
    print("=====================================")


    weather_file_path = "C:/Users/snsur/Desktop/ster/phoenix_az_usa_psm3_tmy-2023.csv"
    location_name = "Phoenix AZ PSM3"
    target_timezone_str = "America/Phoenix"


    module_name = 'Canadian_Solar_Inc__CS6K_275M'
    inverter_name = 'SMA_America__SB5_0_1SP_US_40__240V_'
    module_database = 'CECMod'
    inverter_database = 'CECInverter'


    surface_tilt_strategy = 'latitude'
    surface_azimuth = 180
    modules_per_string = 10
    strings_per_inverter = 2



    battery_capacity_kwh = 25.0
    initial_soc_percent = 60.0
    min_soc_percent = 20.0
    max_charge_power_kw = 5.0
    max_discharge_power_kw = 7.0
    charge_efficiency = 0.95
    discharge_efficiency = 0.90


    baseload_kw = 0.3
    ev_charge_power_kw = 7.0
    ev_charge_start_hour = 22
    ev_charge_end_hour = 4
    ev_charge_frequency_days = 2


    print(f"\nPV Configuration:")
    print(f"  Weather File: {weather_file_path}")
    print(f"  Location: {location_name}, Timezone: {target_timezone_str}")
    print(f"  Module: {module_name}, Inverter: {inverter_name}")
    print(f"  Layout: {modules_per_string}x{strings_per_inverter} modules, Azimuth={surface_azimuth}")


    pv_data, location_obj = load_pv_generation(
        weather_file=weather_file_path,
        location_name=location_name,
        tz_str=target_timezone_str
    )


    if pv_data is not None and location_obj is not None:
        print("\n-------------------------------------")
        print("Weather Data & Location Loaded")
        print(f"Data points: {len(pv_data)}, Time range: {pv_data.index.min()} to {pv_data.index.max()}")
        print("-------------------------------------")


        print("\n--- Defining PV System ---")
        try:
            module_params = retrieve_sam(module_database)[module_name]
            inverter_params = retrieve_sam(inverter_database)[inverter_name]
            print(f"Found Module & Inverter parameters in SAM databases.")
        except KeyError as e:
            print(f"\n*** ERROR: Cannot find '{e}' in SAM database. Check names/availability. ***"); exit()

        temp_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        tilt = location_obj.latitude if surface_tilt_strategy == 'latitude' else float(surface_tilt_strategy)
        print(f"System Geometry: Tilt={tilt:.2f} deg, Azimuth={surface_azimuth} deg")

        system = PVSystem(surface_tilt=tilt, surface_azimuth=surface_azimuth,
                          module_parameters=module_params, inverter_parameters=inverter_params,
                          temperature_model_parameters=temp_params,
                          modules_per_string=modules_per_string, strings_per_inverter=strings_per_inverter,
                          name=f"{location_name} PV System")
        print(f"PVSystem object '{system.name}' created.")


        print("\n--- Running PV Simulation (ModelChain) ---")
        req_cols = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed']
        if missing := [c for c in req_cols if c not in pv_data.columns]:
            print(f"\n*** ERROR: Missing weather columns for ModelChain: {missing} ***"); exit()


        mc = ModelChain(system, location_obj, aoi_model='ashrae', spectral_model='no_loss', name=system.name + " Chain")
        print(f"ModelChain object created (AOI='ashrae', Spectral='no_loss').")
        print("Running PV simulation...")
        try:
            mc.run_model(pv_data)
            print("PV simulation complete.")
        except Exception as mc_err:
            print(f"\n*** ERROR during ModelChain run: {mc_err} ***"); traceback.print_exc(); exit()


        if hasattr(mc, 'results') and hasattr(mc.results, 'ac') and mc.results.ac is not None:
            print("\n--- Preparing Off-Grid Simulation Inputs ---")
            pv_generation_ac_kw = mc.results.ac.fillna(0) / 1000.0


            total_load_kw = create_load_profile(
                index=pv_generation_ac_kw.index, baseload_kw=baseload_kw,
                ev_kw=ev_charge_power_kw, ev_start=ev_charge_start_hour,
                ev_end=ev_charge_end_hour, ev_freq_days=ev_charge_frequency_days
            )
            print(f"Load profile created (Base: {baseload_kw}kW, EV: {ev_charge_power_kw}kW every {ev_charge_frequency_days} days)")


            battery_params = {
                'capacity_kwh': battery_capacity_kwh,
                'initial_soc_kwh': battery_capacity_kwh * (initial_soc_percent / 100.0),
                'min_soc_kwh': battery_capacity_kwh * (min_soc_percent / 100.0),
                'max_charge_kw': max_charge_power_kw,
                'max_discharge_kw': max_discharge_power_kw,
                'charge_eff': charge_efficiency,
                'discharge_eff': discharge_efficiency,
            }
            print(f"Battery Params: Cap={battery_capacity_kwh:.1f}kWh, Limits={max_charge_power_kw:.1f}kW(C)/{max_discharge_power_kw:.1f}kW(D), Eff={charge_efficiency:.2f}(C)/{discharge_efficiency:.2f}(D)")


            print("\n--- Running Off-Grid Battery Simulation ---")
            try:
                offgrid_results, hours_per_step = simulate_battery(pv_generation_ac_kw, total_load_kw, battery_params)
                print(f"Battery simulation complete (using {hours_per_step:.3f} hours per step).")


                print("\n--- Off-Grid Simulation Summary ---")
                total_unmet_load_kwh = offgrid_results['unmet_kw'].sum() * hours_per_step
                total_curtailment_kwh = offgrid_results['curtail_kw'].sum() * hours_per_step
                min_soc_reached_percent = offgrid_results['soc_percent'].min()
                final_soc_percent = offgrid_results['soc_percent'].iloc[-1]
                avg_soc_percent = offgrid_results['soc_percent'].mean()

                print(f"Total Unmet Load (Energy): {total_unmet_load_kwh:.2f} kWh")
                print(f"Total Curtailed PV Energy: {total_curtailment_kwh:.2f} kWh")
                print(f"Battery SoC Range: {min_soc_reached_percent:.1f}% (min) to {offgrid_results['soc_percent'].max():.1f}% (max)")
                print(f"Average Battery SoC: {avg_soc_percent:.1f}%")
                print(f"Final Battery SoC: {final_soc_percent:.1f}%")

                if total_unmet_load_kwh > 0.1:
                    unmet_hours = (offgrid_results['unmet_kw'] > 1e-3).sum()
                    print(f"\n*** WARNING: Load was unmet for approx {unmet_hours} hours. System may be undersized. ***")
                else:
                    print("\nSUCCESS: All load was met by the PV and battery system for the simulated period.")



            except Exception as batt_sim_err:
                print(f"\n*** ERROR during battery simulation: {batt_sim_err} ***"); traceback.print_exc()

        else:
            print("\n*** ERROR: PV simulation AC results not found. Cannot proceed with off-grid simulation. ***")

    else:
        print("\n-------------------------------------")
        print("Failed to load weather data or location. Cannot proceed.")
        print("-------------------------------------")

    print("\n=====================================")
    print("Script Finished.")
    print("=====================================")