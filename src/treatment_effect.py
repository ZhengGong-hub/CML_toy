from mcf.mcf_functions import ModifiedCausalForest
from mcf.reporting import McfOptPolReport
import matplotlib
import pandas as pd
import json
import os
import shutil

def run_treatment_effect_analysis(df):
    """
    Run treatment effect analysis using ModifiedCausalForest on the input dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the preprocessed data
        
    Returns:
        tuple: Results from the analysis and the MCF model
    """
    # Remove output directories if they exist
    for dir_name in ['output', 'out']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

    def compute_salary_outcomes(df):
        """Compute salary outcomes for each period."""
        for period in range(3, 10):
            earnings_cols = [f'EARNX{period}_{quarter}' for quarter in range(1, 5)]
            df[f'SAL_{period}'] = 3 * df[earnings_cols].sum(axis=1)
        df['SAL_AVG'] = df[[f'SAL_{period}' for period in range(3, 10)]].mean(axis=1)
        return df

    def compute_employment_outcomes(df):
        """Compute employment outcomes including total quarters and changes."""
        # Get all employment columns
        empl_cols = [f'EMPLX{period}_{quarter}' 
                    for period in range(3, 10) 
                    for quarter in range(1, 5)]
        
        # Calculate total quarters of employment
        df['EMPL_TTL'] = df[empl_cols].sum(axis=1)
        
        # Calculate employment changes
        is_employed = df[empl_cols].eq(1)
        transitions_to_employed = is_employed & ~is_employed.shift(axis=1).fillna(False)
        transitions_from_employed = ~is_employed & is_employed.shift(axis=1).fillna(False)
        df['EMPL_CHGE'] = transitions_to_employed.sum(axis=1) + transitions_from_employed.sum(axis=1)
        
        return df

    # Compute all outcomes
    df = compute_salary_outcomes(df)
    df = compute_employment_outcomes(df)

    # Split data into training and prediction sets
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = len(df_shuffled) // 2
    training_df = df_shuffled.iloc[:split_idx]
    prediction_df = df_shuffled.iloc[split_idx:]

    # import parameter json
    with open('src/parameter.json', 'r') as f:
        parameter = json.load(f)
    
    # Parameters of the ModifiedCausalForest
    VAR_D_NAME = parameter['treatment']  # Name of treatment variable
    VAR_Y_NAME = ['SAL_AVG'] + ['SAL_'+str(i) for i in range(3,10)] + ['EMPL_TTL'] + ['EMPL_CHGE'] # Name of outcome variables
    VAR_X_NAME_ORD = parameter['ord_covariates']
    VAR_X_NAME_UNORD = parameter['unord_covariates']
    VAR_Z_NAME_ORD = parameter['ord_Z']
    VAR_Z_NAME_UNORD = parameter['unord_Z']

    mymcf = ModifiedCausalForest(
        var_d_name=VAR_D_NAME,
        var_y_name=VAR_Y_NAME,
        var_x_name_ord=VAR_X_NAME_ORD,
        var_x_name_unord=VAR_X_NAME_UNORD,
        var_z_name_ord=VAR_Z_NAME_ORD,
        var_z_name_unord=VAR_Z_NAME_UNORD,
        _int_show_plots=False,
        gen_output_type=2
        )

    matplotlib.use('Agg') # to avoid that plots show up and stop the execution 
    mymcf.train(training_df)
    results, _ = mymcf.predict(prediction_df) 
    
    try:
        results_with_cluster_id_df, _ = mymcf.analyse(results)
    except TypeError:
        pass
    
    my_report = McfOptPolReport(mcf=mymcf, outputfile='Modified-Causal-Forest_Report')
    my_report.report()
    print('End of computations.')
    return results, mymcf