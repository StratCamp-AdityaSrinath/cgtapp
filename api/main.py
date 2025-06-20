from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from io import StringIO

# The Flask app instance
app = Flask(__name__)
CORS(app) # This enables your frontend to call the backend

# --- DATA: Your proprietary data remains secure on the backend ---
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
10,Libmeldy,Metachromatic Leukodystrophy,1.0,1,10,0,10,Total Population,0.000002,0.000004,4250000
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

def run_full_simulation(drug_keys_to_include, sample_size, pp_deductible, agg_deductible, elig_share, uptake, undiag_prev):
    disease_data_full = pd.read_csv(StringIO(DISEASE_DATA_STRING))
    population_data = pd.read_csv(StringIO(POPULATION_DATA_STRING))
    
    disease_data = disease_data_full[disease_data_full['Key'].isin(drug_keys_to_include)].copy()
    if disease_data.empty: return {"error": "No therapies selected."}

    population_data['Age_Value'] = population_data['Age'].str.split('-').str[0].str.replace('+', '').astype(int)
    us_population_total = population_data['Total_Population'].sum()
    n_iterations = 1000

    lambdas_inc, lambdas_prev = [], []
    pop_age_indexed = population_data.set_index('Age_Value')
    pop_totals = {
        'Total Population': population_data['Total_Population'].sum(),
        'Male': population_data['Male_Population'].sum(),
        'Female': population_data['Female_Population'].sum()
    }

    for _, row in disease_data.iterrows():
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
        
        lambda_inc = commercially_relevant_cases_inc * (sample_size / us_population_total)
        lambda_prev = commercially_relevant_cases_prev * (sample_size / us_population_total)
        
        lambdas_inc.append(lambda_inc)
        lambdas_prev.append(lambda_prev)

    lambdas_inc, lambdas_prev = np.array(lambdas_inc), np.array(lambdas_prev)
    cost_mean, loa_values, p_survive = disease_data['Admin_Cost_2025'].values, disease_data['LOA'].values, 0.98

    results = []
    for i in range(n_iterations):
        is_approved = np.random.rand(len(disease_data)) < loa_values
        claims_prev = np.random.poisson(lambdas_prev * p_survive * elig_share * uptake) * is_approved
        claims_inc = np.random.poisson(lambdas_inc * (1 + undiag_prev) * elig_share * uptake) * is_approved
        cost_prev = claims_prev * np.maximum(0, cost_mean - pp_deductible)
        cost_inc = claims_inc * np.maximum(0, cost_mean - pp_deductible)
        total_cost_pre_agg = np.sum(cost_prev) + np.sum(cost_inc)
        cost_after_agg = np.maximum(0, total_cost_pre_agg - agg_deductible)
        pmpm_total = cost_after_agg / sample_size / 12 if sample_size > 0 else 0
        cost_prev_after_deduct, cost_inc_after_deduct = np.sum(cost_prev), np.sum(cost_inc)
        total_cost_after_deduct = cost_prev_after_deduct + cost_inc_after_deduct
        pmpm_prev = pmpm_total * (cost_prev_after_deduct / total_cost_after_deduct) if total_cost_after_deduct > 0 else 0
        pmpm_inc = pmpm_total * (cost_inc_after_deduct / total_cost_after_deduct) if total_cost_after_deduct > 0 else 0
        results.append({'pmpm_total': pmpm_total, 'pmpm_prev': pmpm_prev, 'pmpm_inc': pmpm_inc})
        
    results_df = pd.DataFrame(results)

    def calculate_stats(df, col_name, lambda_sum):
        mean, std = df[col_name].mean(), df[col_name].std()
        
        if mean < 1e-9:
            if lambda_sum > 1e-9:
                return {"mean": "$0.0000", "cv": "Extreme", "max_mean": "Extreme"}
            else:
                return {"mean": "$0.0000", "cv": "0.0%", "max_mean": "0.0%"}
        
        cv = (std / mean) if mean > 0 else 0
        max_mean = (df[col_name].max() / mean) if mean > 0 else 0
        return {"mean": f"${mean:.4f}", "cv": f"{cv:.1%}", "max_mean": f"{max_mean:.1%}"}

    return {
        "total": calculate_stats(results_df, 'pmpm_total', lambdas_inc.sum() + lambdas_prev.sum()),
        "prevalence": calculate_stats(results_df, 'pmpm_prev', lambdas_prev.sum()),
        "incidence": calculate_stats(results_df, 'pmpm_inc', lambdas_inc.sum()),
    }

# This is the main API handler function that Vercel will run.
@app.route('/', methods=['POST'])
def handler():
    # 1. Get the user's inputs from the incoming request
    data = request.get_json()

    # 2. Extract the variables from the data
    drug_keys = data.get('drug_keys', [])
    sample_size = data.get('sample_size', 100000)
    pp_deductible = data.get('pp_deductible', 0)
    agg_deductible = data.get('agg_deductible', 0)
    elig_share = data.get('elig_share', 0.75)
    uptake = data.get('uptake', 0.80)
    undiag_prev = data.get('undiag_prev', 0.10)

    # 3. Call your existing simulation function with these inputs
    results = run_full_simulation(
        drug_keys, 
        sample_size, 
        pp_deductible, 
        agg_deductible, 
        elig_share, 
        uptake, 
        undiag_prev
    )

    # 4. Return ONLY the final results as JSON
    return jsonify(results)

# This part is only for running the server locally for testing.
# Vercel does not use this when deployed.
if __name__ == "__main__":
    app.run(debug=True, port=5001)
