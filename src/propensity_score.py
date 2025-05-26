from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
def propensity_score(df):
    """
    Calculate propensity scores for each treatment type and visualize their distributions.
    
    This function:
    1. Prepares features for propensity score modeling
    2. Fits a multinomial logistic regression model
    3. Generates propensity scores for each treatment type
    4. Creates visualization plots comparing propensity score distributions
    5. Saves the plots to output files
    
    Args:
        df (pandas.DataFrame): Input dataframe containing treatment and covariate data
    """

    # import parameter json
    with open('src/parameter.json', 'r') as f:
        parameter = json.load(f)
    
    # Define features for propensity score model
    X = parameter['ord_covariates'] + parameter['unord_covariates']
    T = parameter['treatment']
    
    df_x = df[X]
    df_t = df[T]
    
    # Fit propensity score model
    ps_model = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(df_x, df_t)
    
    # Generate propensity scores for each treatment type
    df_ps_0 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 0])
    df_ps_1 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 1])
    df_ps_2 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 2])
        
    # Create visualizations
    treatment_labels = ['Non Treated', 'Training Program 1', 'Training Program 2']
    df_ps_list = [df_ps_0, df_ps_1, df_ps_2]
    
    # Plot propensity scores by subsamples
    create_propensity_plot(df_ps_list, treatment_labels, by_subsample=True, 
                          filename='output_data/propensity_score_subsample.png')
    
    # Plot propensity scores for whole sample
    create_propensity_plot(df_ps_list, treatment_labels, by_subsample=False,
                          filename='output_data/propensity_score_whole_sample.png')


def create_propensity_plot(df_ps_list, treatment_labels, by_subsample=True, filename=None):
    """
    Create and save propensity score distribution plots.
    
    Args:
        df_ps_list (list): List of dataframes with propensity scores for each treatment type
        treatment_labels (list): Labels for each treatment type
        by_subsample (bool): If True, plot distributions by treatment subsamples
        filename (str): Path to save the output figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    for i, (df_ps, ax) in enumerate(zip(df_ps_list, axes)):
        if by_subsample:
            # Plot propensity score distributions by treatment subsamples
            for j, label in enumerate(treatment_labels):
                sns.histplot(
                    df_ps.query(f"PTYPE == {j}")["propensity_score"], 
                    kde=True, 
                    label=f"Subsample of {label}", 
                    bins=30, 
                    ax=ax
                )
        else:
            # Plot propensity score distribution for whole sample
            sns.histplot(
                df_ps["propensity_score"], 
                kde=True, 
                bins=30, 
                label="Whole Sample", 
                ax=ax
            )
        
        ax.set_title(f'Propensity Score for {treatment_labels[i]}')
        ax.legend()
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')