from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import pandas as pd
import numpy as np
from io import StringIO
import sys # Import the sys module to print to stderr for Vercel logs

# The Flask app instance. Vercel's handler will look for this 'app' object.
app = Flask(__name__)
CORS(app) # This enables your frontend to call the backend

# --- DATA ---
DISEASE_DATA_STRING = """
Key,Drug_IDs,Indication,LOA,Median_Age_Diagnosis,Median_Life_Expectancy,Age_Min,Age_Max,Segment,Incidence_2025,Prevalence_2025,Admin_Cost_2025
1,Zolgensma,Spinal Muscular Atrophy,1.0,0,2,0,2,Total Population,0.00000015,0.0000003,2100000
2,Luxturna,Inherited Retinal Disease,1.0,20,50,3,99,Total Population,0.0000002,0.0000006,850000
3,Skysona,Cerebral Adrenoleukodystrophy,1.0,7,20,4,17,Male,0.000004,0.00001,3000000
4,Casgevy,Sickle Cell Disease,1.0,25,45,12,64,Total Population,0.000028,0.0003,2200000
5,Lyfgenia,Sickle Cell Disease,1.0,25,45,12,64,Total Population,0.000028,0.0003,3100000
6,Hemgenix,Hemophilia B,1.0,30,60,18,99,Male,0.000005,0.00002,3500000
7,Roctavian,Hemophilia A,1.0,30,60,18,99,Male,0.00002,0.00008,2900000
8,Beqvez,Hemophilia B,0.9,30,60,18,99,Male,0.000005,0.00002,3500000
9,Lenmeldy,Metachromatic Leukodystrophy,1.0,1,10,0,10,Total Population,0.000002,0.000004,4250000
"""

POPULATION_DATA_STRING = """
Age,Total_Population,Male_Population,Female_Population
0-4,19600000,10000000,9600000
5-9,20000000,10200000,9800000
10-14,21000000,10700000,10300000
15-19,21000000,10700000,10300000
20-24,22000000,11200000,10800000
25-29,23000000,11700000,11300000
30-34,24000000,12200000,11800000
35-39,23000000,11700000,11300000
40-44,22000000,11200000,10800000
45-49,21000000,10700000,10300000
50-54,21000000,10700000,10300000
55-59,22000000,11200000,10800000
60-64,21000000,10700000,10300000
65-69,18000000,8500000,9500000
70-74,15000000,7000000,8000000
75-79,10000000,4500000,5500000
80-84,7000000,3000000,4000000
85+,5000000,2000000,3000000
"""

def run_full_simulation(drug_keys_to_include, sample_size, pp_deductible, agg_deductible, therapy_assumptions):
    # --- DEBUG LOG 1: Check inputs to this function ---
    print("--- run_full_simulation ---", file=sys.stderr)
    print(f"DEBUG: drug_keys_to_include received: {drug_keys_to_include}", file=sys.stderr)
    print(f"DEBUG: Type of drug_keys_to_include: {type(drug_keys_to_include)}", file=sys.stderr)
    
    disease_data_full = pd.read_csv(StringIO(DISEASE_DATA_STRING))
    population_data = pd.read_csv(StringIO(POPULATION_DATA_STRING))
    
    # --- DEBUG LOG 2: Check the master data and its key ---
    print(f"DEBUG: disease_data_full 'Key' column dtype: {disease_data_full['Key'].dtype}", file=sys.stderr)
    print(f"DEBUG: disease_data_full 'Key' values: {disease_data_full['Key'].tolist()}", file=sys.stderr)
    
    # --- The point of failure is likely here ---
    disease_data = disease_data_full[disease_data_full['Key'].isin(drug_keys_to_include)].copy()
    
    # --- DEBUG LOG 3: Check the result of the filtering operation ---
    print(f"DEBUG: Shape of disease_data after filtering: {disease_data.shape}", file=sys.stderr)
    
    if disease_data.empty: 
        return {"error": "No therapies were selected or received by the server."}

    population_data['Age_Value'] = population_data['Age'].str.split('-').str[0].str.replace('+', '').astype(int)
    us_population_total = population_data['Total_Population'].sum()
    n_iterations = 1000

    lambdas = {}
    pop_age_indexed = population_data.set_index('Age_Value')
    pop_totals = {
        'Total Population': population_data['Total_Population'].sum(),
        'Male': population_data['Male_Population'].sum(),
        'Female': population_data['Female_Population'].sum()
    }
    for _, row in disease_data.iterrows():
        key = row['Key']
        segment, denominator_pop = row['Segment'], pop_totals.get(row['Segment'], us_population_total)
        total_nat_cases_inc = row['Incidence_2025'] * denominator_pop
        total_nat_cases_prev = row['Prevalence_2025'] * denominator_pop
        age_min, age_max, median_age = row['Age_Min'], row['Age_Max'], row['Median_Age_Diagnosis']
        all_ages = pop_age_indexed.index.values
        sigma, mu = 0.6, np.log(median_age) if median_age > 0 else 0
        with np.errstate(divide='ignore'): log_ages = np.log(all_ages)
        log_ages[all_ages == 0] = 0
        weights = np.exp(-((log_ages - mu)**2) / (2 * sigma**2))
        if np.sum(weights) > 0: weights /= np.sum(weights)
        cases_per_age_inc = total_nat_cases_inc * weights
        cases_per_age_prev = total_nat_cases_prev * weights
        commercial_age_mask = (all_ages >= age_min) & (all_ages <= age_max)
        commercially_relevant_cases_inc = np.sum(cases_per_age_inc[commercial_age_mask])
        commercially_relevant_cases_prev = np.sum(cases_per_age_prev[commercial_age_mask])
        lambdas[key] = {
            'inc': commercially_relevant_cases_inc * (sample_size / us_population_total),
            'prev': commercially_relevant_cases_prev * (sample_size / us_population_total)
        }

    results = []
    for i in range(n_iterations):
        total_cost_prev, total_cost_inc = 0, 0
        for _, row in disease_data.iterrows():
            drug_key_int = row['Key']
            assumptions = therapy_assumptions[drug_key_int]
            if np.random.rand() >= row['LOA']: continue
            p_survive, elig_share, uptake, undiag_prev = 0.98, assumptions['elig_share'], assumptions['uptake'], assumptions['undiag_prev']
            lambda_prev, lambda_inc = lambdas[row['Key']]['prev'], lambdas[row['Key']]['inc']
            claims_prev = np.random.poisson(lambda_prev * p_survive * elig_share * uptake)
            claims_inc = np.random.poisson(lambda_inc * (1 + undiag_prev) * elig_share * uptake)
            cost = row['Admin_Cost_2025']
            total_cost_prev += claims_prev * np.maximum(0, cost - pp_deductible)
            total_cost_inc += claims_inc * np.maximum(0, cost - pp_deductible)

        total_cost_pre_agg = total_cost_prev + total_cost_inc
        cost_after_agg = np.maximum(0, total_cost_pre_agg - agg_deductible)
        pmpm_total = cost_after_agg / sample_size / 12 if sample_size > 0 else 0
        total_cost_after_deduct = total_cost_prev + total_cost_inc
        pmpm_prev = pmpm_total * (total_cost_prev / total_cost_after_deduct) if total_cost_after_deduct > 0 else 0
        pmpm_inc = pmpm_total * (total_cost_inc / total_cost_after_deduct) if total_cost_after_deduct > 0 else 0
        results.append({'pmpm_total': pmpm_total, 'pmpm_prev': pmpm_prev, 'pmpm_inc': pmpm_inc})
        
    results_df = pd.DataFrame(results)

    def calculate_stats(df, col_name, lambda_sum):
        mean, std = df[col_name].mean(), df[col_name].std()
        if mean < 1e-9:
            return {"mean": "$0.0000", "cv": "Extreme", "max_mean": "Extreme"} if lambda_sum > 1e-9 else {"mean": "$0.0000", "cv": "0.0%", "max_mean": "0.0%"}
        cv = (std / mean) if mean > 0 else 0
        max_mean = (df[col_name].max() / mean) if mean > 0 else 0
        return {"mean": f"${mean:.4f}", "cv": f"{cv:.1%}", "max_mean": f"{max_mean:.1%}"}

    total_lambda_inc = sum(l['inc'] for k, l in lambdas.items())
    total_lambda_prev = sum(l['prev'] for k, l in lambdas.items())

    return {
        "total": calculate_stats(results_df, 'pmpm_total', total_lambda_inc + total_lambda_prev),
        "prevalence": calculate_stats(results_df, 'pmpm_prev', total_lambda_prev),
        "incidence": calculate_stats(results_df, 'pmpm_inc', total_lambda_inc),
    }

@app.route('/api/main', methods=['POST'])
def handle_simulation():
    # Using print with sys.stderr ensures logs appear on Vercel
    print("--- handle_simulation ---", file=sys.stderr)
    
    # --- DEBUG LOG 4: Check the raw request body ---
    print(f"DEBUG: Raw request body: {request.data}", file=sys.stderr)
    
    data = request.get_json()
    
    # --- DEBUG LOG 5: Check the parsed JSON data ---
    print(f"DEBUG: Parsed JSON data: {data}", file=sys.stderr)
    
    if data is None:
        return jsonify({"error": "Failed to decode JSON. Please check request format."}), 400

    therapy_assumptions = data.get('therapy_assumptions', {})
    if not therapy_assumptions:
        return jsonify({"error": "No 'therapy_assumptions' provided in the request."}), 400

    # --- DEBUG LOG 6: Check the assumptions dictionary before key conversion ---
    print(f"DEBUG: therapy_assumptions before conversion: {therapy_assumptions}", file=sys.stderr)
    
    therapy_assumptions_int_keys = {int(k): v for k, v in therapy_assumptions.items()}
    drug_keys = list(therapy_assumptions_int_keys.keys())

    # --- DEBUG LOG 7: Check the final list of keys being sent to the simulation ---
    print(f"DEBUG: Final drug_keys list to be used for filtering: {drug_keys}", file=sys.stderr)
    
    results = run_full_simulation(
        drug_keys_to_include=drug_keys,
        sample_size=data.get('sample_size', 100000),
        pp_deductible=data.get('pp_deductible', 0),
        agg_deductible=data.get('agg_deductible', 0),
        therapy_assumptions=therapy_assumptions_int_keys
    )
    return jsonify(results)