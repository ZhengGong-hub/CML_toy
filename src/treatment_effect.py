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
    for dir_name in ['output_treatment_effect', 'output_treatment_effect_placebo']:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)

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
    VAR_Y_NAME = parameter['outcome_variables']
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
        gen_output_type=2,
        gen_outpath='output_treatment_effect'
        )

    matplotlib.use('Agg') # to avoid that plots show up and stop the execution 
    mymcf.train(training_df)
    results, _ = mymcf.predict(prediction_df) 
    
    try:
        results_with_cluster_id_df, _ = mymcf.analyse(results)
    except TypeError:
        pass
    
    my_report = McfOptPolReport(mcf=mymcf, outputfile='Modified-Causal-Forest_Report', outputpath='output_treatment_effect')
    my_report.report()
    print('End of computations.')


    # palcebo test 
    # the idea is to check if the treatment effect has effect on past earnings
    # if it does, then the treatment effect is not due to the fact that the program is effective
    # we hope that the treatment effect is not due to past earnings
    mymcf = ModifiedCausalForest(
        var_d_name=VAR_D_NAME,
        var_y_name=['EARN_X0'],
        var_x_name_ord=VAR_X_NAME_ORD,
        var_x_name_unord=VAR_X_NAME_UNORD,
        var_z_name_ord=VAR_Z_NAME_ORD,
        var_z_name_unord=VAR_Z_NAME_UNORD,
        _int_show_plots=False,
        gen_output_type=2,
        gen_outpath='output_treatment_effect_placebo'
        )

    matplotlib.use('Agg') # to avoid that plots show up and stop the execution 
    mymcf.train(training_df)
    results, _ = mymcf.predict(prediction_df) 
    
    try:
        results_with_cluster_id_df, _ = mymcf.analyse(results)
    except TypeError:
        pass
    
    my_report = McfOptPolReport(mcf=mymcf, outputfile='Modified-Causal-Forest_Report', outputpath='output_treatment_effect_placebo')
    my_report.report()
    print('End of computations placebo test.')
    
    return 