from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

def propensity_score(df):
    """
    Calculate the propensity score for each sample in the dataframe.
    """
    # Create a figure
    df['EARNX1'] =(df['EARNX1_1'] + df['EARNX1_2'] + df['EARNX1_3'] + df['EARNX1_4'])/4
    df['EARNX2'] =(df['EARNX2_1'] + df['EARNX2_2'] + df['EARNX2_3'] + df['EARNX2_4'])/4
    
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

    # variation 1: with subsample
    # Create a figure
    ps_model = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(df_x, df_t)
    df_ps_0 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 0])
    df_ps_1 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 1])
    df_ps_2 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 2])
    print(df_ps_0.head())

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    # First subplot for propensity score 0
    sns.histplot(df_ps_0.query("PTYPE == 0")["propensity_score"], kde=True, label="Subsample of Non Treated", bins=30, ax=axes[0])
    sns.histplot(df_ps_0.query("PTYPE == 1")["propensity_score"], kde=True, label="Subsample of Training Program 1", bins=30, ax=axes[0])
    sns.histplot(df_ps_0.query("PTYPE == 2")["propensity_score"], kde=True, label="Subsample of Training Program 2", bins=30, ax=axes[0])
    axes[0].set_yscale('log')
    axes[0].set_title('Propensity Score for Non Treated')
    axes[0].legend()
    
    # Second subplot for propensity score 1
    sns.histplot(df_ps_1.query("PTYPE == 0")["propensity_score"], kde=True, label="Subsample of Non Treated", bins=30, ax=axes[1])
    sns.histplot(df_ps_1.query("PTYPE == 1")["propensity_score"], kde=True, label="Subsample of Training Program 1", bins=30, ax=axes[1])
    sns.histplot(df_ps_1.query("PTYPE == 2")["propensity_score"], kde=True, label="Subsample of Training Program 2", bins=30, ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Propensity Score for Training Program 1')
    axes[1].legend()
    
    # Third subplot for propensity score 2
    sns.histplot(df_ps_2.query("PTYPE == 0")["propensity_score"], kde=True, label="Subsample of Non Treated", bins=30, ax=axes[2])
    sns.histplot(df_ps_2.query("PTYPE == 1")["propensity_score"], kde=True, label="Subsample of Training Program 1", bins=30, ax=axes[2])
    sns.histplot(df_ps_2.query("PTYPE == 2")["propensity_score"], kde=True, label="Subsample of Training Program 2", bins=30, ax=axes[2])
    axes[2].set_yscale('log')
    axes[2].set_title('Propensity Score for Training Program 2')
    axes[2].legend()

    plt.legend();

    # bar plot of service sector share by region
    # save 
    plt.tight_layout()
    plt.savefig('output_data/propensity_score_subsample.png', dpi=300, bbox_inches='tight')

    # variation 2: no subsample

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    # First subplot for propensity score 0
    sns.histplot(df_ps_0["propensity_score"], kde=True, bins=30, label="Whole Sample", ax=axes[0])
    axes[0].set_title('Propensity Score for Non Treated')
    axes[0].legend()
    
    # Second subplot for propensity score 1
    sns.histplot(df_ps_1["propensity_score"], kde=True, bins=30, label="Whole Sample", ax=axes[1])
    axes[1].set_title('Propensity Score for Training Program 1')
    axes[1].legend()
    
    # Third subplot for propensity score 2
    sns.histplot(df_ps_2["propensity_score"], kde=True, bins=30, label="Whole Sample", ax=axes[2])
    axes[2].set_title('Propensity Score for Training Program 2')
    axes[2].legend()

    plt.legend();

    # bar plot of service sector share by region
    # save 
    plt.tight_layout()
    plt.savefig('output_data/propensity_score_whole_sample.png', dpi=300, bbox_inches='tight')