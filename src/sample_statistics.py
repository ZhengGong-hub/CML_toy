def sample_statistics(df):
    # ptype 0, no program
    df_ptype_0 = df[df['PTYPE'] == 0]
    # ptype 1, program 1
    df_ptype_1 = df[df['PTYPE'] == 1]
    # ptype 2, program 2
    df_ptype_2 = df[df['PTYPE'] == 2]

    # Calculate mean and std for each ptype
    mean_ptype_0 = df_ptype_0.mean()
    std_ptype_0 = df_ptype_0.std()
    mean_ptype_1 = df_ptype_1.mean()
    std_ptype_1 = df_ptype_1.std()
    mean_ptype_2 = df_ptype_2.mean()
    std_ptype_2 = df_ptype_2.std()
    
    # output the results under output_data/statistics.txt in a columnar format
    with open("output_data/sample_statistics.txt", "w") as f:
        # Write header
        f.write(f"{'Variable':<15}{'PTYPE 0':<20}{'PTYPE 1':<20}{'PTYPE 2':<20}\n")
        f.write(f"{'-'*15}{'-'*20}{'-'*20}{'-'*20}\n")
        
        # Get all column names from the dataframe
        columns = df.columns
        
        # Write mean values for each variable
        f.write(f"{'MEANS:':<15}\n")
        for col in columns:
            f.write(f"{col:<15}{mean_ptype_0[col]:<20.4f}{mean_ptype_1[col]:<20.4f}{mean_ptype_2[col]:<20.4f}\n")
        
        # Write std values for each variable
        f.write(f"\n{'STD DEVIATIONS:':<15}\n")
        for col in columns:
            f.write(f"{col:<15}{std_ptype_0[col]:<20.4f}{std_ptype_1[col]:<20.4f}{std_ptype_2[col]:<20.4f}\n")
