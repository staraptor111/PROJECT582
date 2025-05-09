# PROJECT582
This project employs a multi-stage Python workflow to simulate EV integration, model battery degradation, analyze energy balance, and ultimately assess the financial viability of EV charging infrastructure, with a special focus on comparing urban and rural deployments.


Overall Flow:
The process can be broadly categorized into the following interconnected stages:


Battery Degradation Model Development (my_actual_degradation_model.py):
Objective: To create a predictive model for battery health (capacity fade and resistance increase).

Process:
Data Generation: A simulation function, driven by Monte Carlo sampling of various operational parameters (temperature, Depth of Discharge, charge/discharge rates, initial State-of-Health (SoH), traffic levels, etc.) from defined statistical distributions, generates a large dataset. This dataset maps diverse operating conditions to their corresponding degradation outcomes.

Machine Learning: A Machine Learning model (e.g., Neural Network or Tree-based model as specified in the configuration) is trained on this synthetic dataset.
Output: The script saves the trained degradation model (e.g., .keras or .joblib file) and the data scaler used during preprocessing. These become critical inputs for later simulation stages.
Key Outputs: Trained ML model, data scaler, performance plots (Actual vs. Predicted, Feature Importance, Sensitivity).

(Offline Prerequisite) PyBaMM Thermal Analysis (Conceptual - generate_pybamm_lookup.py )
Objective: To provide realistic internal battery operating temperatures based on ambient conditions.
Process: Detailed PyBaMM simulations are run offline across a range of ambient temperatures and potentially load conditions to model the battery's thermal behavior.
Key Output: A lookup table (pybamm_temperature_lookup.csv) mapping ambient temperature to effective internal battery temperature.

City-Scale Energy Balance & Financial Simulation (renewbalanceX.py - e.g., renewbalance7.py):

Objective: To simulate the hourly energy dynamics of a city integrating a growing EV fleet with renewable energy sources and optional grid storage, and to perform an initial financial assessment.
Data Ingestion: Loads city data (population, temperature), renewable energy capacity forecasts (yearly MW for solar/wind per city), hourly renewable generation profiles, and the PyBaMM temperature lookup table.

EV Fleet & Demand Forecasting: Projects EV fleet size based on population and adoption rates. Calculates hourly charging demand, factoring in:

Battery Degradation: Imports and utilizes the trained ML degradation model (from stage 1) and the data scaler. The effective battery temperature from the PyBaMM lookup is a key input here. The model updates the fleet's average SoH over the simulation years.
Renewable Supply Modeling: Calculates hourly renewable energy generation based on interpolated yearly capacities and normalized hourly profiles

Hourly Energy Balance Simulation: For each hour over several years;
Balances EV charging demand with available renewable supply, grid storage (if configured), and grid imports/exports, respecting defined limits.
Tracks key metrics like unmet demand, renewable curtailment, grid draw, and storage state of charge.

Financial Calculation: Based on simulated energy flows, CAPEX/OPEX assumptions, grid electricity prices, and discount rates, it calculates:
Annual system costs.

Levelized Cost of Energy (LCOE) for the city-scale system.
Potential lifetime savings compared to a baseline.
Key Outputs: Aggregate summary CSV per city, detailed hourly simulation results (Parquet/CSV), and various analytical/comparison plots.

Interactive Degradation Prediction (deg_upd_simp.py):
Objective: To provide a user-friendly interface for quick, individual EV battery degradation predictions.
Process: Loads the pre-trained degradation model and scaler (from stage 1). 
takes simplified user inputs (e.g., climate, usage style) and maps them to the model's technical features to generate a prediction.
Key Output: Degradation prediction printed to the console.

Application to EV Charging Station Viability Analysis (Conceptual Framework leveraging outputs from renewbalanceX.py):
Objective: To extend the city-scale analysis to perform detailed financial viability assessments for individual EV charging stations, with a specific focus on comparing urban vs. rural scenarios.
Process (Conceptual - This part describes how the outputs of previous scripts are used for a subsequent, more focused analysis, which might be done in a new script or a detailed spreadsheet model):
Leverage City-Scale Insights: Uses outputs from renewbalanceX.py (regional EV demand patterns, renewable intermittency, grid conditions, battery degradation trends, regional LCOE) as contextual data.
Station-Specific Modeling:
Inputs: Detailed parameters for a specific station (location type: urban/rural, CAPEX/OPEX for chargers/solar/BESS, grid connection costs, land costs). Localized demand forecasts (refined from city-scale data). Revenue stream assumptions (tariffs, V2G, ancillary services). Policy/subsidy information.

Financial Engine: Projects multi-year cash flows for the station. Calculates station-specific KPIs (NPV, IRR, Payback Period, ROI, LCOE of the charging service).
Uncertainty & Scenario Analysis:
Models distinct "Rural" vs. "Urban" scenarios, varying parameters like traffic, grid costs, and land costs.
Performs sensitivity analysis on key drivers (e.g., electricity prices, EV adoption, charger utilization).
Quantifies the viability gap between urban and rural deployments.

Strategic Outputs: Generates viability assessments, risk profiles, evidence-based policy recommendations (e.g., quantifying support needed for rural stations), and investment guidance.
Key Outcome: A comprehensive understanding of the economic feasibility of specific EV charging station projects and strategies to enhance viability, especially in underserved rural areas.
Execution Flow:
Run my_actual_degradation_model.py first to train and save the battery degradation models and scalers.

Run renewbalanceX.py, which will prompt for city selections. This script imports/uses the degradation model from step 1 and the PyBaMM lookup.
The outputs from renewbalanceX.py (summary CSVs, hourly data) then serve as crucial inputs and contextual information for the subsequent (potentially manual or new-script-based) EV Charging Station Viability Analysis.
deg_upd_simp.py can be run anytime after step 1 for quick individual predictions.
This workflow provides a data-driven pipeline from fundamental battery behavior modeling to complex city-wide energy simulations and finally to targeted financial assessments for EV charging infrastructure.
