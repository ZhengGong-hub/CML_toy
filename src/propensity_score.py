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

    # Create a figure
    ps_model = LogisticRegression(C=1e6).fit(df_x, df_t)

    df_ps_0 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 0])
    df_ps_1 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 1])
    df_ps_2 = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 2])
    print(df_ps_0.head())

    sns.histplot(df_ps_0["propensity_score"], kde=True, label="Non Treated", bins=50)
    sns.histplot(df_ps_1["propensity_score"], kde=True, label="Training Program 1", bins=50)
    sns.histplot(df_ps_2["propensity_score"], kde=True, label="Training Program 2", bins=50)

    plt.title("Positivity Check")
    plt.legend();

    # bar plot of service sector share by region
    # save 
    plt.tight_layout()
    plt.savefig('output_data/propensity_score.png', dpi=300, bbox_inches='tight')