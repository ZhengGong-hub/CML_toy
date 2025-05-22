from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
    # Exclude vocational degree 2 
    #   for calculating propensity scores, and look for common support
    #   (i.e. propensity scores and the distribution of the covariates should not be too different between treatment groups)
    df = df.query("VOC_DEG != 2")

    # Calculate average quarterly earnings for years X1 and X2
    df['EARNX1'] = df[['EARNX1_1', 'EARNX1_2', 'EARNX1_3', 'EARNX1_4']].mean(axis=1)
    df['EARNX2'] = df[['EARNX2_1', 'EARNX2_2', 'EARNX2_3', 'EARNX2_4']].mean(axis=1)
    
    # Define features for propensity score model
    X = [
        'AGE', 'SEX', 'SCHOOL', 'VOC_DEG', 'NATION', 'REGION', 
        'REG_AL', 'REG_PRG', 'REG_SER', 'REG_PRO', 'REG_AGRI',
        'SECT_AL', 'PROF_AL', 'PROF_XL',
        'UNEM_X0', 'OLF_X0', 'EMPL_X0', 'EARN_X0', 
        'EMPLX1_1', 'EMPLX1_2', 'EMPLX1_3', 'EMPLX1_4', 
        'EMPLX2_1', 'EMPLX2_2', 'EMPLX2_3', 'EMPLX2_4', 
        'EARNX1', 'EARNX2', 
        'LMP_CW', 'PROF_XL'
    ]
    
    T = ['PTYPE']
    
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