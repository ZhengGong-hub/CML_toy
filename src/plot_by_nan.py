import json

def plot_by_nan(df):
    """
    idea is to plot the boxplot 
    """

    df_shallow = df.copy()
    # if any column has nan
    nan_data = df_shallow[df_shallow.isna().any(axis=1)]
    print(nan_data)

    no_nan_data = df_shallow.dropna()

    # covariates
    with open('src/parameter.json', 'r') as f:
        parameter = json.load(f)
    X = parameter['covariates']

    # Create a table showing statistics for each covariate
    with open("output_data/nan_statistics.txt", "w") as f:
        # Write header
        f.write(f"{'Covariate':<15}{'Mean (NaN)':<15}{'Mean (No NaN)':<15}|{'Std (NaN)':<15}{'Std (No NaN)':<15}\n")
        f.write(f"{'-'*15}{'-'*15}{'-'*15}|{'-'*15}{'-'*15}\n")
        
        # Write statistics for each covariate
        for x in X:
            # Statistics for data with NaN values
            mean_nan = nan_data[x].mean()
            std_nan = nan_data[x].std()
            
            # Statistics for data without NaN values
            mean_no_nan = no_nan_data[x].mean()
            std_no_nan = no_nan_data[x].std()
            
            f.write(f"{x:<15}{mean_nan:<15.4f}{mean_no_nan:<15.4f}|{std_nan:<15.4f}{std_no_nan:<15.4f}\n")
