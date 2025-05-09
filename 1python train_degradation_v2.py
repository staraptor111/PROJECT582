import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.stats import uniform, norm, randint, loguniform
import seaborn as sns
import os
import joblib
import copy
import traceback

_TF_AVAILABLE = False
_SKLEARN_AVAILABLE = False
_XGB_AVAILABLE = False
_LGBM_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1_l2
    from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
    _TF_AVAILABLE = True
    print("TensorFlow found.")
except ImportError: print("WARNING: TensorFlow not found. NN disabled.")
except Exception as e: print(f"ERROR importing TensorFlow: {e}")
try:
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.inspection import permutation_importance
    _SKLEARN_AVAILABLE = True
    print("Scikit-learn found.")
except ImportError: print("ERROR: Scikit-learn not found. Essential ML disabled.")
except Exception as e: print(f"ERROR importing Scikit-learn: {e}")
try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
    print("XGBoost found.")
except ImportError: print("INFO: XGBoost not found. XGB disabled.")
except Exception as e: print(f"ERROR importing XGBoost: {e}")
try:
    import lightgbm as lgbm
    _LGBM_AVAILABLE = True
    print("LightGBM found.")
except ImportError: print("INFO: LightGBM not found. LGBM disabled.")
except Exception as e: print(f"ERROR importing LightGBM: {e}")

# ==============================================================================
#                               1. Configuration
# ==============================================================================
BATTERY_TYPES = {
    "LiIon": {
        "PYBAMM_MODEL_TYPE": "SPMe", "PYBAMM_PARAMETER_SET": "Chen2020",
        "PYBAMM_DEGRADATION_OPTIONS": {"sei": "reaction limited", "lithium plating": "irreversible", "particle cracking": "none"},
        "INPUT_PARAMETER_DISTRIBUTIONS": {
            "avg_temp_c": norm(loc=35, scale=8), "dod": uniform(loc=0.6, scale=0.35),
            "charge_rate": uniform(loc=0.5, scale=1.0), "discharge_rate": uniform(loc=0.8, scale=0.8),
            "cycles_per_day": norm(loc=1.5, scale=0.5), "initial_soh_percent": norm(loc=95, scale=5),
            "traffic_level": randint(low=0, high=3),
        },
        "COLOR": "blue", "SCALER_FILENAME": "LiIon_scaler_v2.joblib", "MODEL_FILENAME": "LiIon_model_v2"
    },
    "NaIon": {
        "PYBAMM_MODEL_TYPE": "SPMe", "PYBAMM_PARAMETER_SET": "NaIon_Parameters_Placeholder",
        "PYBAMM_DEGRADATION_OPTIONS": {"sei": "solvent-diffusion limited", "particle cracking": "none"},
        "INPUT_PARAMETER_DISTRIBUTIONS": {
            "avg_temp_c": norm(loc=30, scale=7), "dod": uniform(loc=0.55, scale=0.35),
            "charge_rate": uniform(loc=0.4, scale=0.8), "discharge_rate": uniform(loc=0.7, scale=0.7),
            "cycles_per_day": norm(loc=1.2, scale=0.4), "initial_soh_percent": norm(loc=96, scale=4),
            "traffic_level": randint(low=0, high=3),
        },
        "COLOR": "orange", "SCALER_FILENAME": "NaIon_scaler_v2.joblib", "MODEL_FILENAME": "NaIon_model_v2"
    }
}
OUTPUT_LABELS = ["capacity_fade_percent", "resistance_increase_percent"]
PYBAMM_SIMULATION_TARGET_CYCLES = 500
N_SIMULATIONS = 1000
ML_MODEL_TYPE = 'NN'
USE_CROSS_VALIDATION = False
ML_TEST_SET_FRACTION = 0.2; N_CV_FOLDS = 5
CALCULATE_FEATURE_IMPORTANCE = True; PLOT_SENSITIVITY_CURVES = True
PROBABILISTIC_ANALYSIS_ENABLED = True
N_MC_SAMPLES = 10000
if _TF_AVAILABLE:
    ML_VALIDATION_SPLIT_DURING_TRAINING = 0.2; ML_EPOCHS = 100
    ML_BATCH_SIZE = 32; ML_LEARNING_RATE = 0.001; ML_NN_OPTIMIZER = Adam
    ML_HIDDEN_LAYER_SIZES = (128, 64, 64, 32); ML_USE_DROPOUT = True; ML_DROPOUT_RATE = 0.15
    ML_USE_REGULARIZATION = False; ML_L1_REG = 0.01; ML_L2_REG = 0.01
    ML_USE_EARLY_STOPPING = True; ML_EARLY_STOPPING_PATIENCE = 15
if _SKLEARN_AVAILABLE:
    TREE_N_ESTIMATORS = 200; TREE_MAX_DEPTH = None
    TREE_LEARNING_RATE = 0.1; TREE_RANDOM_STATE = 42
BASE_OUTPUT_DIR = "ML_Degradation_Training_v2"
PLOTS_PER_PAGE = 9
SENSITIVITY_CURVE_POINTS = 50

# ==============================================================================
#                        2. Placeholder Simulation
# ==============================================================================
def run_mock_simulation(input_params, pybamm_config):
    model_type = pybamm_config.get('PYBAMM_MODEL_TYPE', 'SPMe'); param_set = pybamm_config.get('PYBAMM_PARAMETER_SET', 'Unknown'); deg_options = pybamm_config.get('PYBAMM_DEGRADATION_OPTIONS', {}); target_cycles = pybamm_config.get('PYBAMM_SIMULATION_TARGET_CYCLES', 500)
    try:
        temp = input_params.get('avg_temp_c', 35); dod = input_params.get('dod', 0.8); c_rate_ch = input_params.get('charge_rate', 1.0); base_c_rate_dch = input_params.get('discharge_rate', 1.0); cycles_per_day = input_params.get('cycles_per_day', 1.5); initial_soh = input_params.get('initial_soh_percent', 100.0); traffic = input_params.get('traffic_level', 0)
        discharge_rate_multiplier = 1.0 + (traffic * 0.15); effective_c_rate_dch = base_c_rate_dch * discharge_rate_multiplier; soh_effect_multiplier = 1.0 + max(0, (100.0 - initial_soh) / 100.0 * 2.0)
        if "NaIon" in param_set: base_fade, base_res, temp_effect_fade_exp, dod_effect_fade_exp, sei_multiplier, plating_effect = 2.5, 6.0, 1.05, 2.1, 1.8, 0
        else: base_fade, base_res, temp_effect_fade_exp, dod_effect_fade_exp, sei_multiplier, plating_effect = 2.0, 5.0, 1.1, 2.2, 1.5, (2.0 if deg_options.get("lithium plating") != "none" else 0)
        base_fade *= soh_effect_multiplier; temp_effect_fade = 0.12 * max(0, temp - 25)**temp_effect_fade_exp; dod_effect_fade = 6.0 * dod**dod_effect_fade_exp; sei_effect = sei_multiplier if deg_options.get("sei") != "none" else 0
        cycle_effect_fade = (0.01 + 0.005 * c_rate_ch + 0.002 * effective_c_rate_dch) * target_cycles * soh_effect_multiplier; noise_fade = np.random.normal(0, 0.6); capacity_fade = base_fade + temp_effect_fade + dod_effect_fade + cycle_effect_fade + sei_effect + plating_effect + noise_fade
        temp_effect_res = 0.06 * max(0, temp - 25)**0.9; rate_effect_res = (2.0 * c_rate_ch + 1.0 * effective_c_rate_dch) * soh_effect_multiplier; base_res *= soh_effect_multiplier; noise_res = np.random.normal(0, 1.0); resistance_increase = base_res + temp_effect_res + rate_effect_res + noise_res
        if np.random.rand() < 0.03: return None
        results = {"capacity_fade_percent": max(0, capacity_fade), "resistance_increase_percent": max(0, resistance_increase)}; return results if all(label in results for label in OUTPUT_LABELS) else None
    except Exception as e: print(f"ERROR in sim ({param_set}): {e}"); traceback.print_exc(); return None

# ==============================================================================
#                            3. Data Generation
# ==============================================================================
def generate_data(n_samples, param_distributions, output_labels, simulation_func, pybamm_config, battery_type_name):
    print(f"\n--- Starting Data Generation ({battery_type_name}, {n_samples} Samples) ---"); start_time = time.time()
    results_list = []; input_param_names = list(param_distributions.keys()); successful_runs = 0; failed_runs = 0
    current_pybamm_config = {k: v for k, v in pybamm_config.items() if k.startswith('PYBAMM_')}
    current_pybamm_config['PYBAMM_SIMULATION_TARGET_CYCLES'] = pybamm_config.get('PYBAMM_SIMULATION_TARGET_CYCLES', 500)
    print(f"Input features being generated: {input_param_names}")

    for i in range(n_samples):
        if (i + 1) % (max(1, n_samples // 20)) == 0: print(f"  ({battery_type_name}) Running simulation {i+1}/{n_samples}...")
        sim_inputs = {}
        current_param_name = "N/A"
        try:
            for name, dist in param_distributions.items():
                current_param_name = name
                if hasattr(dist, 'pmf'): 
                    sampled_val = dist.rvs(1)[0]
                    if name == 'traffic_level':
                       sampled_val = int(round(sampled_val))
                elif hasattr(dist, 'pdf'): 
                    sampled_val = dist.rvs(1)[0]
                else:
                    print(f"Warning: Distribution for {name} (type: {type(dist)}) has neither pmf nor pdf. Attempting rvs() anyway.")
                    sampled_val = dist.rvs(1)[0] 

                if name == 'dod': sampled_val = np.clip(sampled_val, 0.01, 1.0)
                elif 'rate' in name: sampled_val = max(0.01, sampled_val)
                elif 'temp' in name: sampled_val = max(-20, min(60, sampled_val))
                elif name == 'initial_soh_percent': sampled_val = np.clip(sampled_val, 70.0, 100.0)

                sim_inputs[name] = sampled_val
        except Exception as e:
            print(f"ERROR sampling run {i+1} ({battery_type_name}) for param '{current_param_name}' with dist type {type(dist)}: {e}")
            failed_runs += 1
            continue

        sim_output = simulation_func(sim_inputs, current_pybamm_config)
        if sim_output is not None and all(label in sim_output for label in output_labels):
            valid_row = True
            for label in output_labels:
                val = sim_output[label]
                if not isinstance(val, (int, float)) or not np.isfinite(val) or val < 0: valid_row = False; break
            if valid_row: data_row = sim_inputs.copy(); data_row.update(sim_output); results_list.append(data_row); successful_runs += 1
            else: failed_runs += 1
        else: failed_runs += 1

    end_time = time.time(); print(f"--- Data Gen Finished ({battery_type_name}) ---\n  OK: {successful_runs}, Fail: {failed_runs}, Time: {end_time - start_time:.2f}s")
    if not results_list: print(f"ERROR: No successful simulations for {battery_type_name}."); return None
    data_df = pd.DataFrame(results_list); print(f"({battery_type_name}) Dataset size: {data_df.shape}"); return data_df

# ==============================================================================
#             4. Machine Learning Model Building Functions
# ==============================================================================
def build_nn_model(input_dim, output_dim, hidden_layers, learning_rate, optimizer_class, use_dropout=False, dropout_rate=0.2, use_regularization=False, l1_reg=0.0, l2_reg=0.01, model_name_suffix=""):
    if not _TF_AVAILABLE: print("ERROR: TensorFlow not available."); return None
    model = Sequential(name=f"NN_Predictor_{model_name_suffix}")
    model.add(Input(shape=(input_dim,), name="Input_Layer"))
    regularizer = l1_l2(l1=l1_reg, l2=l2_reg) if use_regularization else None
    for i, layer_size in enumerate(hidden_layers):
        if not isinstance(layer_size, int) or layer_size <= 0: raise ValueError(f"Hidden layer sizes must be positive integers. Got: {layer_size}")
        model.add(Dense(layer_size, activation='relu', kernel_regularizer=regularizer, name=f"Hidden_{i+1}"))
        if use_dropout: model.add(Dropout(dropout_rate, name=f"Dropout_{i+1}"))
    model.add(Dense(output_dim, activation='linear', name="Output_Layer"))
    try:
        optimizer = optimizer_class(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
        print(f"\n--- NN Model Architecture Built ({model_name_suffix}) ---")
        return model
    except Exception as e: print(f"ERROR compiling NN model ({model_name_suffix}): {e}"); return None

def build_tree_model(model_type, n_estimators, max_depth, learning_rate, random_state, model_name_suffix=""):
    if not _SKLEARN_AVAILABLE: return None; common_params = {'n_estimators': n_estimators, 'random_state': random_state}
    if max_depth is not None: common_params['max_depth'] = max_depth; model = None
    try:
        if model_type == 'RF': model = RandomForestRegressor(**common_params)
        elif model_type == 'GBT': model = GradientBoostingRegressor(learning_rate=learning_rate, **common_params)
        elif model_type == 'XGB':
            if not _XGB_AVAILABLE: print("ERROR: XGBoost missing."); return None
            model = xgb.XGBRegressor(learning_rate=learning_rate, objective='reg:squarederror', **common_params)
        elif model_type == 'LGBM':
            if not _LGBM_AVAILABLE: print("ERROR: LightGBM missing."); return None
            model = lgbm.LGBMRegressor(learning_rate=learning_rate, max_depth=-1 if max_depth is None else max_depth, **common_params)
        else: raise ValueError(f"Bad tree type: {model_type}")
        print(f"\n--- Build {model_type} ({model_name_suffix}) ---");
        if len(OUTPUT_LABELS) > 1: model = MultiOutputRegressor(model, n_jobs=-1); print(f"INFO: Wrapped {model_type} in MultiOutputRegressor.")
        return model
    except Exception as e: print(f"ERROR building {model_type} model ({model_name_suffix}): {e}"); return None

# ==============================================================================
#                           5. Plotting Functions
# ==============================================================================
def _ensure_dir_exists(output_dir):
    try: os.makedirs(output_dir, exist_ok=True); return True
    except OSError as e: print(f"ERROR creating dir {output_dir}: {e}"); return False

def plot_data_distributions(df, output_dir, file_prefix="run", battery_type_tag="", plots_per_page=9):
    title=f"Data Distributions ({battery_type_tag})"
    if df is None or df.empty: print(f"WARN: No data for plot_data_distributions ({battery_type_tag})."); return
    print(f"\n--- Plotting {title} (Prefix: {file_prefix}) ---");
    if not _ensure_dir_exists(output_dir): return
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols: print(f"WARN: No numeric columns to plot ({battery_type_tag})."); return
    n_cols_to_plot = len(numeric_cols); num_figures = int(np.ceil(n_cols_to_plot / plots_per_page))
    for fig_num in range(num_figures):
        start_index = fig_num * plots_per_page; end_index = min((fig_num + 1) * plots_per_page, n_cols_to_plot)
        cols_on_page = numeric_cols[start_index:end_index]; num_plots_on_page = len(cols_on_page)
        if num_plots_on_page == 0: continue
        n_cols_subplot = min(3, num_plots_on_page); n_rows_subplot = int(np.ceil(num_plots_on_page / n_cols_subplot))
        fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(5 * n_cols_subplot, 4 * n_rows_subplot), squeeze=False)
        fig.suptitle(f"{title} (Page {fig_num+1})", fontsize=16, y=0.98); axes_flat = axes.flatten()
        for i, col in enumerate(cols_on_page):
            ax = axes_flat[i]; plot_data = df[col].dropna()
            if plot_data.empty: ax.set_title(f"No Data: {col}")
            else: sns.histplot(plot_data, kde=True, ax=ax, bins=30); ax.set_title(f"Dist: {col}"); ax.set_xlabel(col); ax.grid(axis='y', linestyle='--', alpha=0.7)
        for j in range(num_plots_on_page, len(axes_flat)): axes_flat[j].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = os.path.join(output_dir, f"{file_prefix}_{battery_type_tag}_data_dist_page_{fig_num+1}.png")
        try: fig.savefig(filename, bbox_inches='tight'); print(f"  Saved: {os.path.abspath(filename)}")
        except Exception as e: print(f"ERR save {filename}: {e}")
        finally: plt.close(fig)

def plot_training_history(history, output_dir, file_prefix="run", battery_type_tag="", title_suffix=""):
    title=f"NN Training History ({title_suffix})"
    if history is None or not hasattr(history, 'history') or not history.history: print(f"INFO: No NN history for {battery_type_tag}."); return
    if ML_MODEL_TYPE != 'NN': return
    print(f"\n--- Plotting {title} (Prefix: {file_prefix}) ---");
    if not _ensure_dir_exists(output_dir): return
    history_dict = history.history; has_loss = 'loss' in history_dict; has_val_loss = 'val_loss' in history_dict; has_mae = 'mean_absolute_error' in history_dict; has_val_mae = 'val_mean_absolute_error' in history_dict
    if not has_loss and not has_mae: print(f"WARN: NN history missing loss/mae ({battery_type_tag})."); return
    n_plots = (1 if has_loss else 0) + (1 if has_mae else 0); fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6), squeeze=False)
    fig.suptitle(title, fontsize=16, y=0.98); plot_idx = 0; epochs = range(1, len(history_dict.get('loss', history_dict.get('mean_absolute_error', []))) + 1)
    if has_loss: ax = axes[0, plot_idx]; ax.plot(epochs, history_dict['loss'], label='Train Loss (MSE)', marker='.'); (has_val_loss and ax.plot(epochs, history_dict['val_loss'], label='Val Loss (MSE)', marker='.')); ax.set_title('Loss (MSE)/Epoch'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (MSE)'); ax.legend(); ax.grid(True); ax.set_yscale('log'); plot_idx += 1
    if has_mae: ax = axes[0, plot_idx]; ax.plot(epochs, history_dict['mean_absolute_error'], label='Train MAE', marker='.'); (has_val_mae and ax.plot(epochs, history_dict['val_mean_absolute_error'], label='Val MAE', marker='.')); ax.set_title('MAE/Epoch'); ax.set_xlabel('Epoch'); ax.set_ylabel('MAE'); ax.legend(); ax.grid(True); plot_idx += 1
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); filename = os.path.join(output_dir, f"{file_prefix}_{battery_type_tag}_nn_train_hist.png")
    try: fig.savefig(filename, bbox_inches='tight'); print(f"  Saved: {os.path.abspath(filename)}")
    except Exception as e: print(f"ERR save {filename}: {e}")
    finally: plt.close(fig)

def plot_actual_vs_predicted(y_actual, y_predicted, labels, output_dir, file_prefix="run", battery_type_tag="", model_name="", plots_per_page=6):
    title="Actual vs. Predicted"
    if y_actual is None or y_predicted is None or y_actual.shape != y_predicted.shape or y_actual.ndim != 2 or y_actual.shape[1] == 0: print(f"WARN: Bad data/dims for plot_actual_vs_predicted ({battery_type_tag})."); return
    plot_title = f"{title} ({model_name} - {battery_type_tag})" if model_name else f"{title} ({battery_type_tag})"
    print(f"\n--- Plotting {plot_title} (Prefix: {file_prefix}) ---");
    if not _ensure_dir_exists(output_dir): return
    num_labels_data = y_actual.shape[1]; labels = [f"Out_{i+1}" for i in range(num_labels_data)] if num_labels_data != len(labels) else labels
    n_cols_to_plot = num_labels_data; num_figures = int(np.ceil(n_cols_to_plot / plots_per_page))
    for fig_num in range(num_figures):
        start_index = fig_num * plots_per_page; end_index = min((fig_num + 1) * plots_per_page, n_cols_to_plot); labels_on_page = labels[start_index:end_index]; num_plots_on_page = len(labels_on_page)
        if num_plots_on_page == 0: continue
        n_cols_subplot = min(3, num_plots_on_page); n_rows_subplot = int(np.ceil(num_plots_on_page / n_cols_subplot)); fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(6 * n_cols_subplot, 5.5 * n_rows_subplot), squeeze=False); fig.suptitle(f"{plot_title} (Page {fig_num+1})", fontsize=16, y=0.98); axes_flat = axes.flatten()
        for i_page in range(num_plots_on_page):
            ax = axes_flat[i_page]; label_idx = start_index + i_page; current_label = labels[label_idx]; actual = y_actual[:, label_idx]; predicted = y_predicted[:, label_idx]; valid_indices = np.isfinite(actual) & np.isfinite(predicted)
            if not np.any(valid_indices): ax.set_title(f"{current_label}\n(No valid pts)"); continue
            actual_valid = actual[valid_indices]; predicted_valid = predicted[valid_indices]; r2 = r2_score(actual_valid, predicted_valid) if np.var(actual_valid) > 1e-9 else np.nan
            min_val, max_val = min(actual_valid.min(), predicted_valid.min()), max(actual_valid.max(), predicted_valid.max()); data_range = max_val - min_val; buffer = data_range * 0.1 if data_range > 1e-9 else 1.0; plot_min, plot_max = min_val - buffer, max_val + buffer
            ax.scatter(actual_valid, predicted_valid, alpha=0.4, s=15, label="Data"); ax.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', lw=2, label='Ideal'); ax.set_xlabel(f"Actual {current_label}"); ax.set_ylabel(f"Predicted {current_label}"); title_txt = f"{current_label}" + (f"\n$R^2 = {r2:.3f}$" if not np.isnan(r2) else ""); ax.set_title(title_txt); ax.grid(True, linestyle='--', alpha=0.7); ax.legend(); ax.set_xlim(plot_min, plot_max); ax.set_ylim(plot_min, plot_max); ax.set_aspect('equal', adjustable='box')
        for j in range(num_plots_on_page, len(axes_flat)): axes_flat[j].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); filename = os.path.join(output_dir, f"{file_prefix}_{battery_type_tag}_actual_vs_pred_{model_name}_page_{fig_num+1}.png")
        try: fig.savefig(filename, bbox_inches='tight'); print(f"  Saved: {os.path.abspath(filename)}")
        except Exception as e: print(f"ERR save {filename}: {e}")
        finally: plt.close(fig)

def plot_feature_importance(importance_df, output_dir, file_prefix="run", battery_type_tag="", model_type="", n_features=25):
    if importance_df is None or importance_df.empty: print(f"WARN: No data for plot_feature_importance ({battery_type_tag})."); return
    plot_title = f"{model_type} Feature Importance ({battery_type_tag}, Top {n_features})" if model_type else f"Feat Importance ({battery_type_tag}, Top {n_features})"
    print(f"\n--- Plotting {plot_title} (Prefix: {file_prefix}) ---");
    if not _ensure_dir_exists(output_dir): return
    importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False).head(n_features); fig, ax = plt.subplots(figsize=(10, max(6, int(0.4 * len(importance_df_sorted)))))
    sns.barplot(x='Importance', y='Feature', data=importance_df_sorted, palette='viridis', ax=ax); ax.set_title(plot_title); ax.set_xlabel('Importance Score'); ax.set_ylabel('Feature'); ax.grid(axis='x', linestyle='--', alpha=0.7)
    for container in ax.containers: ax.bar_label(container, fmt='%.3f', fontsize=8, padding=3)
    fig.tight_layout(); filename = os.path.join(output_dir, f"{file_prefix}_{battery_type_tag}_{model_type}_feat_import.png")
    try: fig.savefig(filename, bbox_inches='tight'); print(f"  Saved: {os.path.abspath(filename)}")
    except Exception as e: print(f"ERR save {filename}: {e}")
    finally: plt.close(fig)

def plot_sensitivity_curves(model, scaler, X_train_orig, input_feature_names, output_labels, output_dir, file_prefix, battery_type_tag="", model_name="", n_points=50, plots_per_page=9):
    if model is None or scaler is None or X_train_orig is None or X_train_orig.shape[0] == 0: print(f"WARN: Missing model/scaler/data for sensitivity ({battery_type_tag})."); return
    plot_title_base = f"{model_name} Sensitivity ({battery_type_tag})" if model_name else f"Sensitivity ({battery_type_tag})"
    print(f"\n--- Plotting {plot_title_base} (Prefix: {file_prefix}) ---");
    if not _ensure_dir_exists(output_dir): return
    n_features = len(input_feature_names); n_outputs = len(output_labels);
    try: means = np.nanmean(X_train_orig, axis=0)
    except Exception as mean_err: print(f"ERROR calculating mean for sensitivity ({battery_type_tag}): {mean_err}"); return
    num_figures = int(np.ceil(n_features / plots_per_page))
    for fig_num in range(num_figures):
        start_index = fig_num * plots_per_page; end_index = min((fig_num + 1) * plots_per_page, n_features); features_on_page = input_feature_names[start_index:end_index]; num_plots_on_page = len(features_on_page)
        if num_plots_on_page == 0: continue
        n_cols_subplot = min(3, num_plots_on_page); n_rows_subplot = int(np.ceil(num_plots_on_page / n_cols_subplot)); fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(7 * n_cols_subplot, 5 * n_rows_subplot), squeeze=False); fig.suptitle(f"{plot_title_base} (Page {fig_num+1})", fontsize=16, y=0.98); axes_flat = axes.flatten()
        for i_page, feature_name in enumerate(features_on_page):
            ax = axes_flat[i_page]; feature_index = input_feature_names.index(feature_name)
            try:
                 min_val = np.nanmin(X_train_orig[:, feature_index]); max_val = np.nanmax(X_train_orig[:, feature_index]); sweep_range = np.linspace(min_val, max_val, n_points) if abs(max_val - min_val) > 1e-6 else np.array([min_val])
                 current_means = np.nan_to_num(means); synthetic_data = np.tile(current_means, (len(sweep_range), 1)); synthetic_data[:, feature_index] = sweep_range
                 synthetic_data_scaled = scaler.transform(synthetic_data); predictions = model.predict(synthetic_data_scaled, verbose=0)
                 if predictions.ndim == 1: predictions = predictions.reshape(-1, 1)
                 for output_idx, output_label in enumerate(output_labels): ax.plot(sweep_range, predictions[:, output_idx], label=output_label, marker='.', markersize=4)
                 ax.set_xlabel(f"{feature_name} (Swept)"); ax.set_ylabel("Predicted Output"); ax.set_title(f"Pred vs. {feature_name}"); ax.grid(True, linestyle='--', alpha=0.7); (n_outputs > 1 and ax.legend())
            except IndexError: print(f"ERROR: Feature index {feature_index} ({feature_name}) OOB in sensitivity."); ax.set_title(f"ERR: {feature_name} (Index)")
            except Exception as plot_err: print(f"ERR plot sensitivity {feature_name} ({battery_type_tag}): {plot_err}"); ax.set_title(f"ERR: {feature_name}"); ax.text(0.5, 0.5, f"Plot Error:\n{plot_err}", wrap=True)
        for j in range(num_plots_on_page, len(axes_flat)): axes_flat[j].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); filename = os.path.join(output_dir, f"{file_prefix}_{battery_type_tag}_sensitivity_{model_name}_page_{fig_num+1}.png")
        try: fig.savefig(filename, bbox_inches='tight'); print(f"  Saved: {os.path.abspath(filename)}")
        except Exception as e: print(f"ERR save {filename}: {e}")
        finally: plt.close(fig)

def plot_comparison_sensitivity_curves(models_dict, scalers_dict, data_dict, input_feature_names, output_labels, comparison_output_dir, file_prefix, battery_configs, n_points=50, plots_per_page=9):
    battery_types = list(models_dict.keys())
    if len(battery_types) < 2: print("WARN: Need >= 2 models for comparison sensitivity plot."); return
    print(f"\n--- Plotting Comparison Sensitivity Curves (Prefix: {file_prefix}) ---");
    if not _ensure_dir_exists(comparison_output_dir): return
    first_batt_type = battery_types[0];
    if first_batt_type not in data_dict or 'X_train_orig' not in data_dict[first_batt_type] or data_dict[first_batt_type]['X_train_orig'] is None or data_dict[first_batt_type]['X_train_orig'].shape[0] == 0: print(f"WARN: Reference data missing for {first_batt_type}. Skip comparison."); return
    X_train_orig_ref = data_dict[first_batt_type]['X_train_orig']; n_features = len(input_feature_names); n_outputs = len(output_labels)
    try: means_ref = np.nanmean(X_train_orig_ref, axis=0)
    except Exception as mean_err: print(f"ERROR calculating mean for comparison sensitivity: {mean_err}"); return
    num_figures = int(np.ceil(n_features / plots_per_page))
    for fig_num in range(num_figures):
        start_index = fig_num * plots_per_page; end_index = min((fig_num + 1) * plots_per_page, n_features); features_on_page = input_feature_names[start_index:end_index]; num_plots_on_page = len(features_on_page)
        if num_plots_on_page == 0: continue
        n_cols_subplot = min(3, num_plots_on_page); n_rows_subplot = int(np.ceil(num_plots_on_page / n_cols_subplot)); fig, axes = plt.subplots(n_rows_subplot, n_cols_subplot, figsize=(8 * n_cols_subplot, 6 * n_rows_subplot), squeeze=False); fig.suptitle(f"Comparison Sensitivity Curves (Page {fig_num+1})", fontsize=16, y=0.98); axes_flat = axes.flatten()
        for i_page, feature_name in enumerate(features_on_page):
            ax = axes_flat[i_page]; feature_index = input_feature_names.index(feature_name)
            try:
                 min_val = np.nanmin(X_train_orig_ref[:, feature_index]); max_val = np.nanmax(X_train_orig_ref[:, feature_index]); sweep_range = np.linspace(min_val, max_val, n_points) if abs(max_val - min_val) > 1e-6 else np.array([min_val])
                 current_means_ref = np.nan_to_num(means_ref); synthetic_data_base = np.tile(current_means_ref, (len(sweep_range), 1)); synthetic_data_base[:, feature_index] = sweep_range
                 for batt_type in battery_types:
                     model = models_dict.get(batt_type); scaler = scalers_dict.get(batt_type); config = battery_configs.get(batt_type, {}); batt_color = config.get("COLOR", "gray")
                     if model is None or scaler is None: continue
                     synthetic_data_scaled = scaler.transform(synthetic_data_base); predictions = model.predict(synthetic_data_scaled, verbose=0)
                     if predictions.ndim == 1: predictions = predictions.reshape(-1, 1)
                     for output_idx, output_label in enumerate(output_labels): line_style = '-' if output_idx == 0 else '--'; ax.plot(sweep_range, predictions[:, output_idx], label=f"{batt_type} - {output_label}", marker='.', markersize=3, color=batt_color, linestyle=line_style)
                 ax.set_xlabel(f"{feature_name} (Swept)"); ax.set_ylabel("Predicted Output"); ax.set_title(f"Pred vs. {feature_name}"); ax.grid(True, linestyle='--', alpha=0.7); ax.legend(fontsize='small')
            except IndexError: print(f"ERROR: Feature index {feature_index} ({feature_name}) OOB in comparison plot."); ax.set_title(f"ERR: {feature_name} (Index)")
            except Exception as plot_err: print(f"ERR plot comparison sensitivity {feature_name}: {plot_err}"); ax.set_title(f"ERR: {feature_name}"); ax.text(0.5, 0.5, f"Plot Error:\n{plot_err}", wrap=True)
        for j in range(num_plots_on_page, len(axes_flat)): axes_flat[j].axis('off')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); filename = os.path.join(comparison_output_dir, f"{file_prefix}_comparison_sensitivity_page_{fig_num+1}.png")
        try: fig.savefig(filename, bbox_inches='tight'); print(f"  Saved Comparison Plot: {os.path.abspath(filename)}")
        except Exception as e: print(f"ERR save comparison plot {filename}: {e}")
        finally: plt.close(fig)

def plot_probabilistic_histogram(predictions, label, output_dir, file_prefix, battery_type_tag):
    if not _ensure_dir_exists(output_dir) or predictions is None or len(predictions) == 0: return
    try:
        mean_pred = np.mean(predictions); std_pred = np.std(predictions); ci_95 = np.percentile(predictions, [2.5, 97.5])
        plt.figure(figsize=(8, 5)); sns.histplot(predictions, kde=True, bins=50); plt.title(f"Distribution of Predicted {label} ({battery_type_tag})\n(Mean={mean_pred:.2f}, Std={std_pred:.2f})"); plt.xlabel(f"Predicted {label}"); plt.ylabel("Frequency / Density"); plt.axvline(mean_pred, color='red', linestyle='--', label=f'Mean: {mean_pred:.2f}'); plt.axvline(ci_95[0], color='grey', linestyle=':', label=f'95% CI Low: {ci_95[0]:.2f}'); plt.axvline(ci_95[1], color='grey', linestyle=':', label=f'95% CI High: {ci_95[1]:.2f}'); plt.legend(); plt.grid(axis='y', alpha=0.5)
        label_fname = label.replace('%','percent').replace(' ','_').replace('/','_').lower()
        prob_filename = os.path.join(output_dir, f"{file_prefix}_{battery_type_tag}_probabilistic_hist_{label_fname}.png")
        plt.savefig(prob_filename, bbox_inches='tight'); print(f"    Probabilistic Histogram saved: {os.path.abspath(prob_filename)}")
    except Exception as e: print(f"    ERR saving histogram: {e}")
    finally: plt.close()

# ==============================================================================
#                              6. Main Execution
# ==============================================================================
if __name__ == "__main__":
    main_start_time = time.time()
    print("--- Starting Battery Degradation Prediction Script (Train/Comparison Mode - V2 Fixed Again) ---")
    run_timestamp = time.strftime("%Y%m%d_%H%M%S"); print(f"Run Timestamp: {run_timestamp}")
    if not _SKLEARN_AVAILABLE: print("CRITICAL: Scikit-learn missing. Exit."); exit(1)
    if ML_MODEL_TYPE == 'NN' and not _TF_AVAILABLE: print("ERROR: NN selected, TF missing. Exit."); exit(1)
    if ML_MODEL_TYPE == 'XGB' and not _XGB_AVAILABLE: print("ERROR: XGB selected, XGBoost missing. Exit."); exit(1)
    if ML_MODEL_TYPE == 'LGBM' and not _LGBM_AVAILABLE: print("ERROR: LGBM selected, LightGBM missing. Exit."); exit(1)
    if USE_CROSS_VALIDATION: print("INFO: Running in Cross-Validation mode. Final models/scalers NOT saved.")
    else: print("INFO: Running in Train/Test Split mode. Final models/scalers WILL be saved.")

    main_output_dir = f"{BASE_OUTPUT_DIR}_{run_timestamp}"; abs_main_output_dir = os.path.abspath(main_output_dir)
    if not _ensure_dir_exists(main_output_dir): exit(1)
    print(f"Base output directory: '{abs_main_output_dir}'")

    all_data = {}; all_models = {}; all_scalers = {}; all_training_data = {}

    for batt_type, config in BATTERY_TYPES.items():
        print(f"\n{'='*30} Processing: {batt_type} {'='*30}")
        batt_output_dir = os.path.join(main_output_dir, f"{batt_type}_Results"); _ensure_dir_exists(batt_output_dir)
        if "INPUT_PARAMETER_DISTRIBUTIONS" not in config: print(f"ERROR: Missing INPUT_PARAMETER_DISTRIBUTIONS for {batt_type}. Skip."); continue
        input_distributions = config["INPUT_PARAMETER_DISTRIBUTIONS"]; input_feature_names = list(input_distributions.keys())
        print(f"Using features for training {batt_type}: {input_feature_names}")
        pybamm_sim_config = {k: v for k, v in config.items() if k.startswith('PYBAMM_')}; pybamm_sim_config['PYBAMM_SIMULATION_TARGET_CYCLES'] = PYBAMM_SIMULATION_TARGET_CYCLES
        generated_data = generate_data(N_SIMULATIONS, input_distributions, OUTPUT_LABELS, run_mock_simulation, pybamm_sim_config, batt_type)
        if generated_data is None or generated_data.empty: print(f"\nERROR: Data generation failed for {batt_type}. Skip."); continue
        missing_cols = [col for col in input_feature_names if col not in generated_data.columns]
        if missing_cols: print(f"ERROR: Generated data missing columns: {missing_cols}. Check sim. Skip."); continue
        data_filename = os.path.join(batt_output_dir, f"{run_timestamp}_{batt_type}_generated_data_v2.csv")
        try: generated_data.to_csv(data_filename, index=False); print(f"({batt_type}) Data saved: '{os.path.abspath(data_filename)}'"); all_data[batt_type] = generated_data
        except Exception as e: print(f"WARN: Cannot save data CSV for {batt_type}: {e}")
        plot_data_distributions(generated_data, batt_output_dir, file_prefix=run_timestamp, battery_type_tag=batt_type)

        print(f"\n--- Prepare Data ({batt_type}, {ML_MODEL_TYPE}) ---")
        required_cols = input_feature_names + OUTPUT_LABELS
        if not all(c in generated_data.columns for c in required_cols): print(f"ERROR: Missing required cols in final DF for {batt_type}. Skip."); continue
        X_orig = generated_data[input_feature_names].values; y = generated_data[OUTPUT_LABELS].values
        if np.isnan(X_orig).any() or np.isinf(X_orig).any() or np.isnan(y).any() or np.isinf(y).any():
            print(f"WARN ({batt_type}): NaNs/Infs found. Cleaning..."); valid_rows = ~np.isnan(X_orig).any(axis=1) & ~np.isinf(X_orig).any(axis=1) & ~np.isnan(y).any(axis=1) & ~np.isinf(y).any(axis=1)
            X_orig = X_orig[valid_rows]; y = y[valid_rows]; print(f"({batt_type}) Cleaned shape: X={X_orig.shape}, y={y.shape}")
            if X_orig.shape[0] == 0: print(f"ERROR: No valid data after clean for {batt_type}. Skip."); continue
        scaler = StandardScaler(); all_scalers[batt_type] = scaler; model = None; training_history = None; cv_scores = None

        if USE_CROSS_VALIDATION:
            print(f"\n--- CV ({batt_type}, {N_CV_FOLDS}-Fold, {ML_MODEL_TYPE}) ---")
            kfold = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=TREE_RANDOM_STATE)
            res_r2, res_mse, res_mae = [], [], []
            X_scaled = scaler.fit_transform(X_orig); print(f"({batt_type}) Features scaled before CV.")
            all_scalers[batt_type] = scaler; all_training_data[batt_type] = {'X_train_orig': X_orig}
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled, y)):
                print(f"  --- ({batt_type}) Fold {fold+1}/{N_CV_FOLDS} ---")
                X_train_f, X_val_f = X_scaled[train_idx], X_scaled[val_idx]; y_train_f, y_val_f = y[train_idx], y[val_idx]
                f_model = None; f_hist = None; fold_suffix = f"{batt_type}_Fold{fold+1}"
                if ML_MODEL_TYPE == 'NN':
                    f_model = build_nn_model(X_train_f.shape[1], y_train_f.shape[1], ML_HIDDEN_LAYER_SIZES, ML_LEARNING_RATE, ML_NN_OPTIMIZER, ML_USE_DROPOUT, ML_DROPOUT_RATE, ML_USE_REGULARIZATION, ML_L1_REG, ML_L2_REG, model_name_suffix=fold_suffix)
                    if f_model: cb=[TerminateOnNaN()]; (ML_USE_EARLY_STOPPING and cb.append(EarlyStopping(monitor='val_loss', patience=ML_EARLY_STOPPING_PATIENCE, restore_best_weights=True))); f_hist=f_model.fit(X_train_f, y_train_f, epochs=ML_EPOCHS, batch_size=ML_BATCH_SIZE, validation_data=(X_val_f, y_val_f), callbacks=cb, verbose=0)
                else:
                    f_model = build_tree_model(ML_MODEL_TYPE, TREE_N_ESTIMATORS, TREE_MAX_DEPTH, TREE_LEARNING_RATE, TREE_RANDOM_STATE + fold, model_name_suffix=fold_suffix);
                    if f_model: f_model.fit(X_train_f, y_train_f)
                if f_model: y_pred_f = f_model.predict(X_val_f); mse_f=mean_squared_error(y_val_f, y_pred_f); mae_f=mean_absolute_error(y_val_f, y_pred_f); r2_f=r2_score(y_val_f, y_pred_f); res_mse.append(mse_f); res_mae.append(mae_f); res_r2.append(r2_f); print(f"  Fold {fold+1} ({batt_type}): R2={r2_f:.4f}, MSE={mse_f:.4f}, MAE={mae_f:.4f}")
                else: print(f"Fold {fold+1} ({batt_type}) build Fail."); res_mse.append(np.nan); res_mae.append(np.nan); res_r2.append(np.nan)
            cv_scores = {'Mean R2': np.nanmean(res_r2), 'Std R2': np.nanstd(res_r2), 'Mean MSE': np.nanmean(res_mse), 'Std MSE': np.nanstd(res_mse), 'Mean MAE': np.nanmean(res_mae), 'Std MAE': np.nanstd(res_mae)}; print(f"\n--- CV Summary ({batt_type}) ---"); [print(f"  {m}: {v:.4f}") for m, v in cv_scores.items()];
            print(f"\nINFO: CV done for {batt_type}. No final model/scaler saved.")
        else: 
            print(f"\n--- Train/Test Split ({batt_type}, {ML_MODEL_TYPE}) ---")
            X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_orig, y, test_size=ML_TEST_SET_FRACTION, random_state=TREE_RANDOM_STATE)
            X_train_scaled = scaler.fit_transform(X_train_orig); X_test_scaled = scaler.transform(X_test_orig)
            all_scalers[batt_type] = scaler; all_training_data[batt_type] = {'X_train_orig': X_train_orig, 'y_train': y_train}
            print(f"({batt_type}) Features scaled (fit on train). Scaler n_features_in={scaler.n_features_in_}")

            if ML_MODEL_TYPE == 'NN':
                model = build_nn_model(X_train_scaled.shape[1], y_train.shape[1], ML_HIDDEN_LAYER_SIZES, ML_LEARNING_RATE, ML_NN_OPTIMIZER, ML_USE_DROPOUT, ML_DROPOUT_RATE, ML_USE_REGULARIZATION, ML_L1_REG, ML_L2_REG, model_name_suffix=batt_type)
                if model:
                    print(f"\n--- Train NN ({batt_type}) ---"); cb = [TerminateOnNaN()]; (ML_USE_EARLY_STOPPING and cb.append(EarlyStopping(monitor='val_loss', patience=ML_EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)))
                    t_start = time.time(); training_history = model.fit(X_train_scaled, y_train, epochs=ML_EPOCHS, batch_size=ML_BATCH_SIZE, validation_split=ML_VALIDATION_SPLIT_DURING_TRAINING, callbacks=cb, verbose=1); print(f"--- NN Train Done ({batt_type}, {time.time() - t_start:.2f} s) ---")
                    plot_training_history(training_history, batt_output_dir, file_prefix=run_timestamp, battery_type_tag=batt_type, title_suffix=batt_type)
            else: 
                 model = build_tree_model(ML_MODEL_TYPE, TREE_N_ESTIMATORS, TREE_MAX_DEPTH, TREE_LEARNING_RATE, TREE_RANDOM_STATE, model_name_suffix=batt_type)
                 if model: print(f"\n--- Train {ML_MODEL_TYPE} ({batt_type}) ---"); t_start = time.time(); model.fit(X_train_scaled, y_train); print(f"--- {ML_MODEL_TYPE} Train Done ({batt_type}, {time.time() - t_start:.2f} s) ---")

            if model: 
                all_models[batt_type] = model
                print(f"\n--- Evaluate Test Set ({batt_type}) ---");
                try:
                     y_pred = model.predict(X_test_scaled); mse_test=mean_squared_error(y_test, y_pred); mae_test=mean_absolute_error(y_test, y_pred); r2_test=r2_score(y_test, y_pred); print(f"  Test Metrics: MSE={mse_test:.4f}, MAE={mae_test:.4f}, R2(overall)={r2_test:.4f}"); print(f"  R2 per Output:"); [print(f"    - {l}: {r2_score(y_test[:,i], y_pred[:,i]):.4f}") for i, l in enumerate(OUTPUT_LABELS)]
                     plot_actual_vs_predicted(y_test, y_pred, OUTPUT_LABELS, batt_output_dir, file_prefix=run_timestamp, battery_type_tag=batt_type, model_name=ML_MODEL_TYPE)
                except Exception as e: print(f"ERROR eval test set ({batt_type}): {e}")

                if CALCULATE_FEATURE_IMPORTANCE:
                    print(f"\n--- Feature Importance ({batt_type}) ---"); imp_df = None
                    try:
                        imp = None; base_model = model.estimators_[0] if hasattr(model, 'estimators_') else model
                        if ML_MODEL_TYPE in ['RF','GBT','XGB','LGBM'] and hasattr(base_model, 'feature_importances_'): imp = np.mean([est.feature_importances_ for est in model.estimators_], axis=0) if hasattr(model, 'estimators_') else base_model.feature_importances_
                        if imp is None: perm = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=TREE_RANDOM_STATE, n_jobs=-1, scoring='neg_mean_squared_error'); imp = perm.importances_mean
                        if imp is not None and len(imp) == len(input_feature_names): imp_df = pd.DataFrame({'Feature': input_feature_names, 'Importance': imp}); print(imp_df.sort_values('Importance', ascending=False)); plot_feature_importance(imp_df, batt_output_dir, file_prefix=run_timestamp, battery_type_tag=batt_type, model_type=ML_MODEL_TYPE)
                        elif imp is not None: print(f"CRITICAL WARN ({batt_type}): FI len ({len(imp)}) != Feat names len ({len(input_feature_names)}). Skip plot.")
                        else: print(f"WARN ({batt_type}): Could not calculate FI.")
                    except Exception as e: print(f"ERROR calc FI ({batt_type}): {e}")

                if PLOT_SENSITIVITY_CURVES:
                    print(f"\n--- Sensitivity Curves ({batt_type}) ---");
                    try: plot_sensitivity_curves(model, scaler, X_train_orig, input_feature_names, OUTPUT_LABELS, batt_output_dir, run_timestamp, batt_type, ML_MODEL_TYPE, SENSITIVITY_CURVE_POINTS, PLOTS_PER_PAGE)
                    except Exception as e: print(f"ERROR plot sensitivity ({batt_type}): {e}")

                if PROBABILISTIC_ANALYSIS_ENABLED and "INPUT_PARAMETER_DISTRIBUTIONS" in config:
                     print(f"\n--- Probabilistic Analysis ({batt_type}, {N_MC_SAMPLES} samples) ---")
                     prob_distributions = config["INPUT_PARAMETER_DISTRIBUTIONS"]; mc_inputs_list = []; prob_feature_names = list(prob_distributions.keys())
                     for idx_mc in range(N_MC_SAMPLES):
                         sample = {}
                         current_mc_param_name = "N/A"
                         try:
                             for name, dist in prob_distributions.items():
                                 current_mc_param_name = name
                                 if hasattr(dist, 'pmf'): sampled_val = dist.rvs(1)[0]
                                 elif hasattr(dist, 'pdf'): sampled_val = dist.rvs(1)[0]
                                 else: sampled_val = dist.rvs(1)[0] 

                                 if name == 'dod': sampled_val = np.clip(sampled_val, 0.01, 1.0)
                                 elif 'rate' in name: sampled_val = max(0.01, sampled_val)
                                 elif 'temp' in name: sampled_val = max(-20, min(60, sampled_val))
                                 elif name == 'initial_soh_percent': sampled_val = np.clip(sampled_val, 70.0, 100.0)
                                 elif name == 'traffic_level': sampled_val = int(round(sampled_val))
                                 sample[name] = sampled_val
                             mc_inputs_list.append(sample)
                         except Exception as e: print(f"WARN: Error sampling MC run {idx_mc+1} for param '{current_mc_param_name}': {e}")
                     if mc_inputs_list:
                         mc_inputs_df = pd.DataFrame(mc_inputs_list); mc_inputs_array = mc_inputs_df[input_feature_names].values; mc_inputs_scaled = scaler.transform(mc_inputs_array); mc_predictions = model.predict(mc_inputs_scaled, verbose=0)
                         if mc_predictions.ndim == 1: mc_predictions = mc_predictions.reshape(-1, 1)
                         print(f"\n--- Probabilistic Analysis Prediction Distributions ({batt_type}) ---")
                         for i, label in enumerate(OUTPUT_LABELS):
                             try: preds_for_label = mc_predictions[:, i]; print(f"  Output: {label}\n    Mean: {np.mean(preds_for_label):.3f}, StdDev: {np.std(preds_for_label):.3f}, 95% CI: [{np.percentile(preds_for_label, 2.5):.3f}, {np.percentile(preds_for_label, 97.5):.3f}]"); plot_probabilistic_histogram(preds_for_label, label, batt_output_dir, run_timestamp, batt_type)
                             except Exception as e: print(f"ERR plotting probabilistic hist for {label}: {e}")
                     else: print(f"WARN: No valid MC samples generated ({batt_type}).")

                print(f"\n--- Save Scaler & Model ({batt_type}) ---")
                scaler_fname_base = None; model_fname_base=None
                try: 
                    scaler_to_save = all_scalers.get(batt_type)
                    if scaler_to_save and hasattr(scaler_to_save, 'n_features_in_'):
                        scaler_fname_base = config.get("SCALER_FILENAME", f"{batt_type}_scaler_v2.joblib"); scaler_fname_full = os.path.join(batt_output_dir, scaler_fname_base)
                        joblib.dump(scaler_to_save, scaler_fname_full); print(f"({batt_type}) Scaler saved: '{os.path.abspath(scaler_fname_full)}'")
                    else: print(f"CRITICAL WARN ({batt_type}): Scaler not found or not fitted. Cannot save.")
                except Exception as e: print(f"ERROR saving scaler ({batt_type}): {e}")
                try: 
                    model_fname_base = config.get("MODEL_FILENAME", f"{batt_type}_model_v2"); model_extension = '.keras' if ML_MODEL_TYPE == 'NN' else '.joblib'
                    model_fname_ts = os.path.join(batt_output_dir, f"{run_timestamp}_{model_fname_base}_{ML_MODEL_TYPE}{model_extension}")
                    model_fname_simple = os.path.join(batt_output_dir, f"{model_fname_base}{model_extension}")
                    if ML_MODEL_TYPE == 'NN': model.save(model_fname_ts); model.save(model_fname_simple)
                    else: joblib.dump(model, model_fname_ts); joblib.dump(model, model_fname_simple)
                    print(f"({batt_type}) Model saved (timestamped): '{os.path.abspath(model_fname_ts)}'")
                    print(f"({batt_type}) Model also saved as (simple): '{os.path.abspath(model_fname_simple)}'")
                    if scaler_fname_base and model_fname_base: print(f"  INFO: Copy '{model_fname_base}{model_extension}' and '{scaler_fname_base}' to 'pretrained_models_v2'")
                except Exception as e: print(f"ERROR saving model ({batt_type}): {e}")
            else: print(f"\nSkipping evaluation/saving for {batt_type}: model build/train failed.")

        print(f"\n{'='*30} Finished Processing: {batt_type} {'='*30}")

    if len(all_models) > 1 and PLOT_SENSITIVITY_CURVES and not USE_CROSS_VALIDATION:
        print(f"\n{'='*30} Generating Comparison Plots {'='*30}")
        comparison_output_dir = os.path.join(main_output_dir, "Comparison_Plots"); can_plot_comparison = True
        if 'input_feature_names' in locals():
            for batt_type in all_models.keys():
                if batt_type not in all_scalers or not hasattr(all_scalers[batt_type],'n_features_in_'): print(f"WARN: Fitted scaler missing for {batt_type}. No comparison."); can_plot_comparison = False; break
                if batt_type not in all_training_data or 'X_train_orig' not in all_training_data[batt_type]: print(f"WARN: Train data missing for {batt_type}. No comparison."); can_plot_comparison = False; break
            if can_plot_comparison: plot_comparison_sensitivity_curves(all_models, all_scalers, all_training_data, input_feature_names, OUTPUT_LABELS, comparison_output_dir, run_timestamp, BATTERY_TYPES, SENSITIVITY_CURVE_POINTS, PLOTS_PER_PAGE)
            else: print("Skipping comparison plots due to missing components.")
        else: print("WARN: Cannot create comparison sensitivity plots, 'input_feature_names' not found.")
    else: print("\nSkipping comparison plots (check conditions: >1 model, sensitivity enabled, not CV mode).")

    print(f"\n{'='*60}\n--- Script Finished ---\nTotal execution time: {time.time() - main_start_time:.2f} seconds")
    print(f"All results saved in base directory: '{abs_main_output_dir}'")
    if not USE_CROSS_VALIDATION: print("INFO: Remember to copy the SIMPLY NAMED V2 model/scaler files to 'pretrained_models_v2'.\n"+"="*60)