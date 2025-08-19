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

def calculate_stats(df_col, lambda_sum):
    mean, std = df_col.mean(), df_col.std()
    if mean < 1e-9:
        is_extreme = lambda_sum > 1e-9
        return {
            "mean": "$0.0000",
            "cv": "Extreme" if is_extreme else "0.0%",
            "max_mean": "Extreme" if is_extreme else "0.0%"
        }
    cv = (std / mean) if mean > 0 else 0
    max_mean = (df_col.max() / mean) if mean > 0 else 0
    return {"mean": f"${mean:.4f}", "cv": f"{cv:.1%}", "max_mean": f"{max_mean:.1%}"}

def calculate_incidence_stats_and_premiums(df_col, lambda_sum, exp_gross_up, p95_gross_up, p99_gross_up):
    # Base stats
    base_stats = calculate_stats(df_col, lambda_sum)
    
    # VaR Calculations
    p95 = df_col.quantile(0.95)
    p99 = df_col.quantile(0.99)
    p999 = df_col.quantile(0.999)
    base_stats['p95'] = f"${p95:.4f}"
    base_stats['p99'] = f"${p99:.4f}"
    base_stats['p999'] = f"${p999:.4f}"

    # Premium Calculations
    mean_val = df_col.mean()
    
    def get_premium_breakdown(base_pmpm, gross_up_rate):
        profit_load = base_pmpm * (gross_up_rate / 3.0)
        expense_load = base_pmpm * (2.0 * gross_up_rate / 3.0)
        total_premium = base_pmpm + profit_load + expense_load
        return {
            "total": f"${total_premium:.4f}",
            "expense": f"${expense_load:.4f}",
            "profit": f"${profit_load:.4f}"
        }

    base_stats['premiums'] = {
        'expectation': get_premium_breakdown(mean_val, exp_gross_up),
        'p95': get_premium_breakdown(p95, p95_gross_up),
        'p99': get_premium_breakdown(p99, p99_gross_up)
    }
    
    return base_stats


def run_full_simulation(drug_keys_to_include, sample_size, pp_deductible, agg_deductible, therapy_assumptions, exp_gross_up, p95_gross_up, p99_gross_up):
    disease_data_full = pd.read_csv(StringIO(DISEASE_DATA_STRING))
    population_data = pd.read_csv(StringIO(POPULATION_DATA_STRING))
    
    # Filter for selected therapies
    disease_data = disease_data_full[disease_data_full['Key'].isin(drug_keys_to_include)].copy()
    
    if disease_data.empty: 
        return {"error": "No therapies were selected or received by the server."}

    population_data['Age_Value'] = population_data['Age'].str.split('-').str[0].str.replace('+', '').astype(int)
    us_population_total = population_data['Total_Population'].sum()
    n_iterations = 1000

    # Calculate lambdas (expected number of cases)
    lambdas = {}
    pop_age_indexed = population_data.set_index('Age_Value')
    pop_totals = {
        'Total Population': population_data['Total_Population'].sum(),
        'Male': population_data['Male_Population'].sum(),
        'Female': population_data['Female_Population'].sum()
    }
    for _, row in disease_data.iterrows():
        key = row['Key']
        denominator_pop = pop_totals.get(row['Segment'], us_population_total)
        total_nat_cases_inc = row['Incidence_2025'] * denominator_pop
        total_nat_cases_prev = row['Prevalence_2025'] * denominator_pop
        
        # Simplified age weighting for this example
        age_min, age_max = row['Age_Min'], row['Age_Max']
        age_mask = (population_data['Age_Value'] >= age_min) & (population_data['Age_Value'] <= age_max)
        relevant_pop = population_data[age_mask][f"{row['Segment']}_Population"].sum() if row['Segment'] != 'Total Population' else population_data[age_mask]['Total_Population'].sum()
        
        commercially_relevant_cases_inc = total_nat_cases_inc * (relevant_pop / denominator_pop if denominator_pop > 0 else 0)
        commercially_relevant_cases_prev = total_nat_cases_prev * (relevant_pop / denominator_pop if denominator_pop > 0 else 0)
        
        lambdas[key] = {
            'inc': commercially_relevant_cases_inc * (sample_size / us_population_total),
            'prev': commercially_relevant_cases_prev * (sample_size / us_population_total)
        }

    # Monte Carlo Simulation
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
        
        cost_after_pp_deduct = total_cost_prev + total_cost_inc
        pmpm_prev = pmpm_total * (total_cost_prev / cost_after_pp_deduct) if cost_after_pp_deduct > 0 else 0
        pmpm_inc = pmpm_total * (total_cost_inc / cost_after_pp_deduct) if cost_after_pp_deduct > 0 else 0
        results.append({'pmpm_total': pmpm_total, 'pmpm_prev': pmpm_prev, 'pmpm_inc': pmpm_inc})
            
    results_df = pd.DataFrame(results)

    # Calculate final stats
    total_lambda_inc = sum(l['inc'] for k, l in lambdas.items())
    total_lambda_prev = sum(l['prev'] for k, l in lambdas.items())

    return {
        "total": calculate_stats(results_df['pmpm_total'], total_lambda_inc + total_lambda_prev),
        "prevalence": calculate_stats(results_df['pmpm_prev'], total_lambda_prev),
        "incidence": calculate_incidence_stats_and_premiums(
            results_df['pmpm_inc'], total_lambda_inc, exp_gross_up, p95_gross_up, p99_gross_up
        ),
    }

@app.route('/api/main', methods=['POST'])
def handle_simulation():
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Failed to decode JSON. Please check request format."}), 400

    therapy_assumptions = data.get('therapy_assumptions', {})
    if not therapy_assumptions:
        return jsonify({"error": "No 'therapy_assumptions' provided in the request."}), 400

    therapy_assumptions_int_keys = {int(k): v for k, v in therapy_assumptions.items()}
    drug_keys = list(therapy_assumptions_int_keys.keys())
    
    results = run_full_simulation(
        drug_keys_to_include=drug_keys,
        sample_size=data.get('sample_size', 100000),
        pp_deductible=data.get('pp_deductible', 0),
        agg_deductible=data.get('agg_deductible', 0),
        therapy_assumptions=therapy_assumptions_int_keys,
        exp_gross_up=data.get('exp_gross_up', 0.50),
        p95_gross_up=data.get('p95_gross_up', 0.20),
        p99_gross_up=data.get('p99_gross_up', 0.10)
    )
    return jsonify(results)