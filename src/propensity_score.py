from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

def propensity_score(df):
    """
    Calculate the propensity score for each sample in the dataframe.
    """
    # Create a figure
    
    X = [
        'AGE', 'SEX', 'SCHOOL', 'VOC_DEG', 'NATION', 'REGION', 'REG_AL', 'REG_PRG', 
        'SECT_AL', 'PROF_AL', 'UNEM_X0', 'OLF_X0', 'EMPL_X0', 'EARN_X0', 
        'EMPLX1_1', 'EMPLX1_2', 'EMPLX1_3', 'EMPLX1_4', 
        'EMPLX2_1', 'EMPLX2_2', 'EMPLX2_3', 'EMPLX2_4', 
        'EARNX1_1', 'EARNX1_2', 'EARNX1_3', 'EARNX1_4', 
        'EARNX2_1', 'EARNX2_2', 'EARNX2_3', 'EARNX2_4', 
        'LMP_CW', 'PROF_XL'
    ]

    T = ['PTYPE']

    df_x = df[X]
    df_t = df[T]

    # Create a figure
    ps_model = LogisticRegression(C=1e6).fit(df_x, df_t)

    df_ps = df.assign(propensity_score=ps_model.predict_proba(df[X])[:, 1])
    print(df_ps.head())

    sns.distplot(df_ps.query("PTYPE==0")["propensity_score"], kde=True, label="Non Treated")
    sns.distplot(df_ps.query("PTYPE==1")["propensity_score"], kde=True, label="Training Program 1")
    sns.distplot(df_ps.query("PTYPE==2")["propensity_score"], kde=True, label="Training Program 2")

    plt.title("Positivity Check")
    plt.legend();

    # bar plot of service sector share by region
    # save 
    plt.tight_layout()
    plt.savefig('output_data/propensity_score.png', dpi=300, bbox_inches='tight')