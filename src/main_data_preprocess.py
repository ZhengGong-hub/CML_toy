import os
import requests
import pandas as pd
from dotenv import load_dotenv
from sample_statistics import sample_statistics
from plot_ptype import plot_ptype
from propensity_score import propensity_score
from plot_by_region import plot_by_region
from plot_by_nan import plot_by_nan
def load_data(csv_path):
    """
    columns:
    ['PERS', 'AGE', 'SEX', 'SCHOOL', 'VOC_DEG', 'NATION', 'REGION', 'REG_AL', 
    'REG_PRG', 'REG_SER', 'REG_PRO', 'REG_AGRI', 'SECT_AL', 'PROF_AL', 'SPECIA_CW', 
    'SHP_CW_1', 'SHP_CW_2', 'SHP_CW_3', 'SHP_CW_4', 'UNEM_X0', 'OLF_X0', 'EMPL_X0', 
    'EARN_X0', 'EMPLX1_1', 'EMPLX1_2', 'EMPLX1_3', 'EMPLX1_4', 'EMPLX2_1', 'EMPLX2_2', 
    'EMPLX2_3', 'EMPLX2_4', 'EMPLX3_1', 'EMPLX3_2', 'EMPLX3_3', 'EMPLX3_4', 'EMPLX4_1', 
    'EMPLX4_2', 'EMPLX4_3', 'EMPLX4_4', 'EMPLX5_1', 'EMPLX5_2', 'EMPLX5_3', 'EMPLX5_4', 
    'EMPLX6_1', 'EMPLX6_2', 'EMPLX6_3', 'EMPLX6_4', 'EMPLX7_1', 'EMPLX7_2', 'EMPLX7_3', 
    'EMPLX7_4', 'EMPLX8_1', 'EMPLX8_2', 'EMPLX8_3', 'EMPLX8_4', 'EMPLX9_1', 'EMPLX9_2', 
    'EMPLX9_3', 'EMPLX9_4', 'EARNX1_1', 'EARNX1_2', 'EARNX1_3', 'EARNX1_4', 'EARNX2_1', 
    'EARNX2_2', 'EARNX2_3', 'EARNX2_4', 'EARNX3_1', 'EARNX3_2', 'EARNX3_3', 'EARNX3_4', 
    'EARNX4_1', 'EARNX4_2', 'EARNX4_3', 'EARNX4_4', 'EARNX5_1', 'EARNX5_2', 'EARNX5_3', 
    'EARNX5_4', 'EARNX6_1', 'EARNX6_2', 'EARNX6_3', 'EARNX6_4', 'EARNX7_1', 'EARNX7_2', 
    'EARNX7_3', 'EARNX7_4', 'EARNX8_1', 'EARNX8_2', 'EARNX8_3', 'EARNX8_4', 'EARNX9_1', 
    'EARNX9_2', 'EARNX9_3', 'EARNX9_4', 'LMP_CW', 'PTYPE', 'C_T1', 'C_T2', 'C_T3', 
    'C_T4', 'DURAT', 'PROF_XL']

    PERS Individual identifier (case id derived from social security number; same records may
    appear more than once; such duplicate records may be deleted)
    PTYPE Programme type (1: T1; 2: T2; 3: E1; 4: E2; 0: no programme)
    DURAT Duration of programme (planned) in Months
    EARN_X0 Average monthly earnings in the 10 years prior 19X1.
    EARNX1_y Average monthly earnings 19X1, (y) quarter [y: 1, 2, 3, 4], local currency
    EARNX2_y Average monthly earnings 19X2, (y) quarter [y: 1, 2, 3, 4], local currency
    EARNX3_y Average monthly earnings 19X3, (y) quarter [y: 1, 2, 3, 4], local currency
    EARNX4_y Average monthly earnings 19X4, (y) quarter [y: 1, 2, 3, 4], local currency
    EARNX5_y Average monthly earnings 19X5, (y) quarter [y: 1, 2, 3, 4], local currency
    EARNX6_y Average monthly earnings 19X6, (y) quarter [y: 1, 2, 3, 4], local currency
    EARNX7_y Average monthly earnings 19X7, (y) quarter [y: 1, 2, 3, 4], local currency
    EARNX8_y Average monthly earnings 19X8, (y) quarter [y: 1, 2, 3, 4], local currency
    EARNX9_y Average monthly earnings 19X9, (y) quarter [y: 1, 2, 3, 4], local currency
    UNEM_X0 Average number of months of reg. unemployment in 10 years before 19X1
    EM_X0 Average number of months of employment in 10 years before 19X1
    OLF_X0 Average number of months out-of-the-labour-force in 10 years before 19X1
    EMPLX1_y Employment state 19X1, (y) quarter [..], [1: employed; 2: reg. unemployed; 3: neither]
    EMPLX2_y Employment state 19X2, (y) quarter [..], [1: employed; 2: unemployed; 3: neither]
    EMPLX3_y Employment state 19X3, (y) quarter [..], [1: employed; 2: unemployed; 3: neither]
    EMPLX4_y Employment state 19X4, (y) quarter [..], [1: employed; 2: unemployed; 3: neither]
    EMPLX5_y Employment state 19X5, (y) quarter [..], [1: employed; 2: unemployed; 3: neither]
    EMPLX6_y Employment state 19X6, (y) quarter [..], [1: employed; 2: unemployed; 3: neither]
    EMPLX7_y Employment state 19X7, (y) quarter [..], [1: employed; 2: unemployed; 3: neither]
    EMPLX8_y Employment state 19X8, (y) quarter [..], [1: employed; 2: unemployed; 3: neither]
    EMPLX9_y Employment state 19X9, (y) quarter [..], [1: employed; 2: unemployed; 3: neither]
    AGE Age in years
    C_T1 Individual assigned to T1 but for whom the course was cancelled
    C_T2 Individual assigned to T2 but for whom the course was cancelled
    C_E1 Individual assigned to E1 but for whom the course was cancelled
    C_E2 Individual assigned to E2 but for whom the course was cancelled
    SEX Sex (1 male, 0 female)
    SCHOOL Schooling (degrees in years; 8: no degree)
    VOC_DEG Vocational degree (0: None; 1: below university; 2 university)
    NATION Nationality: 1 Local; 2 other European; 3: Asian; 4 African; 5: American
    LMP_CW Labour market prospects without programme as assessed by case worker (1 very
    bad, 4 very good)
    SHP_CW_y Caseworkers share of clients allocated to programme y
    SPECIA_CW Unemployed in contact with caseworker with additional resources for ALMP
    REGION Region of labour office (1-85)
    REG_AL Regional unemployment rate in %
    REG_PRG Regional share of unemployed participating in programmes
    REG_SER Regional share of service sector
    REG_PRO Regional share of production sector
    REG_AGRI Regional share of agriculture
    SECT_AL Unemployment rate in sector of last occupation
    PROF_AL Unemployment rate in profession of last occupation
    PROF_XL Professional unemployment rate (variable not verified by Section XX.12 of Department
    """
    df = pd.read_csv(csv_path)
    return df

def preprocess_data(df):
    # shallow copy df
    df_shallow = df.copy()

    # Record initial sample size
    initial_sample_size = len(df_shallow)

    # STEP 0: calculate the average quarterly earnings for years X1 and X2
    df_shallow['EARNX1'] = df_shallow[['EARNX1_1', 'EARNX1_2', 'EARNX1_3', 'EARNX1_4']].mean(axis=1)
    df_shallow['EARNX2'] = df_shallow[['EARNX2_1', 'EARNX2_2', 'EARNX2_3', 'EARNX2_4']].mean(axis=1)

    # STEP 0: impute We recognized that we excluded missing observations for the following three variables REG_SER, REG_PRO, and REG_AGRL. However these variables sum up to one and are equal for each region. Instead would you could try, if possible, to just infer the missing values from persons in the same region
    # if the missing value is in REG_SER, use the 1 - REG_PRG - REG_AGRI
    # if the missing value is in REG_PRO, use the 1 - REG_PRG - REG_AGRI
    # if the missing value is in REG_AGRI, use the 1 - REG_PRG - REG_SER
    df_shallow['REG_SER'] = df_shallow['REG_SER'].fillna(1 - df_shallow['REG_PRG'] - df_shallow['REG_AGRI'])
    df_shallow['REG_PRO'] = df_shallow['REG_PRO'].fillna(1 - df_shallow['REG_PRG'] - df_shallow['REG_AGRI'])
    df_shallow['REG_AGRI'] = df_shallow['REG_AGRI'].fillna(1 - df_shallow['REG_PRG'] - df_shallow['REG_SER'])

    # STEP 1: drop all the samples that are assigned to a employment program
    # we are only interested in the effect of training program on the outcome
    # so we drop all the columns that are related to employement program, i.e. PTYPE=3 and PTYPE=4
    df_shallow = df_shallow[~df_shallow['PTYPE'].isin([3, 4])]
    after_step1_size = len(df_shallow)
    
    # STEP 2: drop all the samples that are assigned to a cancelled program
    # our instruction did not talk about why and how the assigned program was cancelled
    # so we drop all the samples that are assigned to a cancelled program
    df_shallow = df_shallow[~((df_shallow['C_T1'] == 1) | (df_shallow['C_T2'] == 1) | (df_shallow['C_T3'] == 1) | (df_shallow['C_T4'] == 1))]
    # afterwards, drop all the columns that are related to cancelled program
    df_shallow = df_shallow.drop(columns=['C_T1', 'C_T2', 'C_T3', 'C_T4'])
    after_step2_size = len(df_shallow)

    # STEP 3: record the statistics and drop all the samples that have NaN values
    # Check which columns have NaN values and count them
    nan_counts = df_shallow.isna().sum()
    # Filter to only show columns with at least one NaN value
    columns_with_nans = nan_counts[nan_counts > 0]
    
    # Calculate percentage of NaN values in each column
    nan_percentage = round((columns_with_nans / len(df_shallow)) * 100, 2)
    # Save NaN percentage information to a text file
    
    os.makedirs("output_data", exist_ok=True)
    with open("output_data/nan_percentage.txt", "w") as f:
        for col, value in nan_percentage.items():
            f.write(f"{col:<12} {value}%\n")
    print('NaN percentage information saved to output_data/nan_percentage.txt')
    
    # drop NaN values 
    plot_by_nan(df_shallow)

    df_shallow = df_shallow.dropna()
    final_sample_size = len(df_shallow)

    # STEP 4: drop all samples that have age not in 30-50 
    df_shallow = df_shallow[df_shallow['AGE'].isin(range(30, 51))]
    after_step4_size = len(df_shallow)

    # STEP 5: drop duplicates 
    df_shallow = df_shallow.drop_duplicates(subset=['PERS'], keep=False)
    after_step5_size = len(df_shallow)
    
    # Save sample size information to a text file
    with open("output_data/sample_sizes.txt", "w") as f:
        f.write(f"Initial sample size: {initial_sample_size}\n")
        f.write(f"After removing employment programs: {after_step1_size}\n")
        f.write(f"After removing cancelled programs: {after_step2_size}\n")
        f.write(f"After removing NaN values: {final_sample_size}\n")
        f.write(f"After removing age not in 30-50: {after_step4_size}\n")
        f.write(f"After removing duplicates: {after_step5_size}\n")
        f.write(f"Total samples removed: {initial_sample_size - final_sample_size}\n")
        f.write(f"Percentage of samples retained: {round((final_sample_size / initial_sample_size) * 100, 2)}%\n")
    print('Sample size information saved to output_data/sample_sizes.txt')

    return df_shallow

def check_distribution(df1, df2, column_names):
    # check the distribution of the two dataframes
    # in terms of mean and standard deviation difference in %
    # for all columns
    results = []
    results.append("Distribution comparison, df1 is raw data, df2 is preprocessed data:")
    for col in column_names:
        df1_mean = df1[col].mean()
        df2_mean = df2[col].mean()
        mean_diff = abs(df1_mean - df2_mean)
        mean_diff_pct = (mean_diff / df1_mean) * 100 if df1_mean != 0 else 0
        
        df1_std = df1[col].std()
        df2_std = df2[col].std()
        std_diff = abs(df1_std - df2_std)
        std_diff_pct = (std_diff / df1_std) * 100 if df1_std != 0 else 0
        
        results.append(f"{col}: df1_mean={df1_mean:.4f}, df2_mean={df2_mean:.4f}, mean_diff_pct={mean_diff_pct:.2f}%, df1_std={df1_std:.4f}, df2_std={df2_std:.4f}, std_diff_pct={std_diff_pct:.2f}%")
    
    # Save distribution comparison to a text file
    with open("output_data/distribution_comparison.txt", "w") as f:
        for line in results:
            f.write(f"{line}\n")
    print('Distribution comparison saved to output_data/distribution_comparison.txt')

def main():
    df = load_data("CML_public/West.csv")
    # print(df.columns.tolist())

    df_preprocessed = preprocess_data(df)
    check_distribution(df, df_preprocessed, ['AGE', 'SCHOOL', 'SEX'])

    print(df_preprocessed.head())
    print(df_preprocessed.shape)

    sample_statistics(df_preprocessed)
    propensity_score(df_preprocessed)
    plot_ptype(df_preprocessed)
    plot_by_region(df_preprocessed)

if __name__ == "__main__":
    main() 