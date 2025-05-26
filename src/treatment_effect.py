from mcf.mcf_functions import ModifiedCausalForest
from mcf.reporting import McfOptPolReport
import matplotlib
import pandas as pd

def run_treatment_effect_analysis(df):
    """
    Run treatment effect analysis using ModifiedCausalForest on the input dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the preprocessed data
        
    Returns:
        tuple: Results from the analysis and the MCF model
    """
    # COMPUTE OUTCOME VARIABLES
    # outcome 1: SAL_i, i=3,...,9
    for i in range(3, 10):
        df['SAL_'+str(i)] = 3 * df[['EARNX'+str(i)+'_'+str(j) for j in range(1,5)]].sum(axis=1)
    # outcome 2: SAL_AVG
    df['SAL_AVG'] = df[['SAL_'+str(j) for j in range(3,10)]].mean(axis=1)

    # SPLIT THE DATAFRAME IN 2 DATAFRAMES (1 for prediction, 1 for training)
    # Shuffle the rows randomly
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split in half
    half = len(df_shuffled) // 2
    training_df, prediction_df = df_shuffled.iloc[:half], df_shuffled.iloc[half:]

    # Parameters of the ModifiedCausalForest
    VAR_D_NAME = 'PTYPE'   # Name of treatment variable
    VAR_Y_NAME = 'SAL_9' # Name of outcome variable
    VAR_X_NAME_ORD = [
        "SEX", "SPECIA_CW", "AGE", "SCHOOL", "VOC_DEG", "REG_AL", "REG_SER", "REG_PRO", "REG_AGRI", "SECT_AL", "PROF_AL", 
        "UNEM_X0", "OLF_X0", "EMPL_X0", "EARN_X0", "EMPLX1_1", "EMPLX1_2", "EMPLX1_3", "EMPLX1_4", 
        "EMPLX2_1", "EMPLX2_2", "EMPLX2_3", "EMPLX2_4",	"EARNX1_1", "EARNX1_2", "EARNX1_3", "EARNX1_4", 
        "EARNX2_1", "EARNX2_2", "EARNX2_3", "EARNX2_4", "LMP_CW"
    ]      
    VAR_X_NAME_UNORD = ["NATION", "REGION"] 

    mymcf = ModifiedCausalForest(
                                var_d_name=VAR_D_NAME,
                                var_y_name=VAR_Y_NAME,
                                var_x_name_ord=VAR_X_NAME_ORD,
                                var_x_name_unord=VAR_X_NAME_UNORD, 
                                _int_show_plots=False,
                                gen_output_type=2
                                )

    matplotlib.use('Agg') # to avoid that plots show up and stop the execution 
    mymcf.train(training_df) 
    results, _ = mymcf.predict(prediction_df) 
    
    try: # this is to overcome an error with mymcf.analyse(results) that shows up after plots are generated
        results_with_cluster_id_df, _ = mymcf.analyse(results)
        my_report = McfOptPolReport(mcf=mymcf, outputfile='Modified-Causal-Forest_Report')
        my_report.report()
        print('End of computations.\n\nThanks for using ModifiedCausalForest.'
        ' \n\nYours sincerely\nMCF \U0001F600')
    except TypeError as e:
        my_report = McfOptPolReport(mcf=mymcf, outputfile='Modified-Causal-Forest_Report')
        my_report.report()
        print('End of computations.\n\nThanks for using ModifiedCausalForest.'
            ' \n\nYours sincerely\nMCF \U0001F600')
    
    return results, mymcf