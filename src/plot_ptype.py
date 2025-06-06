import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def set_style():
    """Set the style for publication-quality plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    # Set the style parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 12,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def add_statistical_annotations(ax, df, x_col, y_col, filter_condition=None):
    """Add statistical annotations to boxplots
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add annotations to
    df : pandas.DataFrame
        The dataframe containing the data
    x_col : str
        The column name for the x-axis categories
    y_col : str
        The column name for the y-axis values
    filter_condition : callable, optional
        A function that takes a dataframe and returns a filtered dataframe
    """
    # Apply filter if provided
    data = df if filter_condition is None else filter_condition(df)
    
    # Add statistical annotations for each category
    for i, category in enumerate(sorted(data[x_col].unique())):
        category_data = data[data[x_col] == category][y_col]
        q1 = category_data.quantile(0.25)
        median = category_data.median()
        mean = category_data.mean()
        q3 = category_data.quantile(0.75)
        
        # Position the text next to each boxplot
        ax.text(i + 0.3, median, 
                f'Q1: {q1:.1f}\nMean: {mean:.1f}\nMedian: {median:.1f}\nQ3: {q3:.1f}',
                horizontalalignment='left', verticalalignment='center', 
                size='small', color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

def plot_ptype(df):
    """Create publication-quality plots for program type analysis
    """
    set_style()
    plot_sex_by_ptype(df)
    plot_age_by_ptype(df)
    plot_duration_by_ptype(df)
    plot_ptype_frequency(df)
    plot_ptype_past_income(df)
    plot_ptype_school(df)
    plot_ptype_vocational_degree(df)
    plot_ptype_labour_market_prospects(df)
    plot_ptype_nationality(df)
    plot_ptype_unemployment_rate_last_occupation(df)
    plot_ptype_region(df)
    plot_num_month_unemployed_by_ptype(df)
    plot_num_month_out_of_labour_force_by_ptype(df)

def plot_ptype_frequency(df):
    """Create a plot of the frequency of each program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)
    # plot bar plot of ptype frequency
    ptype_frequency = df['PTYPE'].value_counts()
    sns.barplot(x=ptype_frequency.index, y=ptype_frequency.values, ax=ax1, width=0.5)
    ax1.set_title('Frequency of Each Program Type')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Frequency')
    
    # Add count labels on the bars
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%d', label_type='center')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/ptype_frequency.png', dpi=300, bbox_inches='tight')


def plot_sex_by_ptype(df):
    """Create a plot of the sex distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)
    # 1. Gender Distribution by Program Type
    gender_by_ptype = pd.crosstab(df['PTYPE'], df['SEX'])
    
    # Calculate percentages for each PTYPE
    gender_pct = gender_by_ptype.div(
        gender_by_ptype.sum(axis=1), axis=0) * 100
    
    gender_by_ptype.plot(kind='bar', stacked=True, ax=ax1, 
                        color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Sex Distribution by Program Type', pad=20)
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Count')
    ax1.legend(['1.0', '2.0'], title='Sex')
    
    # Add percentage labels on the bars
    for i, c in enumerate(ax1.containers):
        # Get the percentages
        percentages = gender_pct.iloc[:, i].values
        
        # Create labels with percentage
        labels = [f'{pct:.1f}%' for pct in percentages]
        ax1.bar_label(c, labels=labels, label_type='center')
    
    # Add sample sizes as text
    for i, ptype in enumerate(df['PTYPE'].unique()):
        n = len(df[df['PTYPE'] == ptype])
    
    # Set x-axis tick labels to horizontal (not tilted)
    plt.xticks(rotation=0)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/sex_by_ptype.png', dpi=300, bbox_inches='tight')

def plot_age_by_ptype(df):
    """Create a plot of the age distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)
    
    # plot boxplot of age by ptype with reduced width
    sns.boxplot(x='PTYPE', y='AGE', data=df, ax=ax1, width=0.5)
    
    # Set the title and labels
    ax1.set_title('Age Distribution by Program Type')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Age (AGE)')
    
    # Add statistical annotations
    add_statistical_annotations(ax1, df, 'PTYPE', 'AGE')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/age_by_ptype.png', dpi=300, bbox_inches='tight')

def plot_duration_by_ptype(df):
    """Create a plot of the duration distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # Filter out PTYPE 0 as we don't care about it
    filtered_df = df[df['PTYPE'] != 0]

    # plot boxplot of duration by ptype
    sns.boxplot(x='PTYPE', y='DURAT', data=filtered_df, ax=ax1, width=0.5)

    # Set the title and labels
    ax1.set_title('Duration Distribution by Program Type (excluding PTYPE 0)')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Duration (DURAT)')

    # Add statistical annotations
    add_statistical_annotations(ax1, df, 'PTYPE', 'DURAT', 
                               filter_condition=lambda df: df[df['PTYPE'] != 0])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/duration_by_ptype.png', dpi=300, bbox_inches='tight')

def plot_ptype_past_income(df):
    """Create a plot of the past income distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)
    # plot boxplot of past income by ptype
    sns.boxplot(x='PTYPE', y='EARN_X0', data=df, ax=ax1, width=0.5)

    # Set the title and labels
    ax1.set_title('Past Income Distribution by Program Type')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Past Income (EARN_X0)')

    # Add statistical annotations
    add_statistical_annotations(ax1, df, 'PTYPE', 'EARN_X0')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/ptype_past_income.png', dpi=300, bbox_inches='tight')

def plot_ptype_school(df):
    """Create a plot of the school distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # degree in years, 8: no degree
    # bar plot of degree by ptype
    # Calculate counts by program type and school
    degree_by_ptype = pd.crosstab(df['PTYPE'], df['SCHOOL'])
    
    # Calculate percentages for each PTYPE
    degree_by_ptype_pct = degree_by_ptype.div(
        degree_by_ptype.sum(axis=1), axis=0) * 100
    
    # Plot the counts
    degree_by_ptype.plot(kind='bar', stacked=True, ax=ax1, 
                        color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
    ax1.set_title('Degree Distribution by Program Type (SCHOOL)')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Count')
    
    # Prevent x-axis labels from tilting
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Add count and percentage labels on the bars
    for i, c in enumerate(ax1.containers):
        labels = [f'{degree_by_ptype_pct.iloc[j, i]:.1f}%' 
                 for j, v in enumerate(c.datavalues)]
        ax1.bar_label(c, labels=labels, label_type='center')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/ptype_school.png', dpi=300, bbox_inches='tight')

def plot_ptype_vocational_degree(df):
    """Create a plot of the vocational degree distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # bar plot of vocational degree by ptype
    vocational_degree_by_ptype = pd.crosstab(df['PTYPE'], df['VOC_DEG'])
    
    # Calculate percentages for each PTYPE
    vocational_degree_pct = vocational_degree_by_ptype.div(
        vocational_degree_by_ptype.sum(axis=1), axis=0) * 100
    
    vocational_degree_by_ptype.plot(kind='bar', stacked=True, ax=ax1, 
                                   color=['#2ecc71', '#3498db', '#f39c12'])
    ax1.set_title('Vocational Degree Distribution by Program Type (VOC_DEG)')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Count')

    # Prevent x-axis labels from tilting
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Add percentage labels on the bars
    for i, c in enumerate(ax1.containers):
        # Get the percentages
        percentages = vocational_degree_pct.iloc[:, i].values
        
        # Create labels with percentage
        labels = [f'{pct:.1f}%' for pct in percentages]
        ax1.bar_label(c, labels=labels, label_type='center')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/ptype_vocational_degree.png', dpi=300, bbox_inches='tight')

def plot_ptype_labour_market_prospects(df):
    """Create a plot of the labour market prospects distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # bar plot of labour market prospects by ptype
    labour_market_prospects_by_ptype = pd.crosstab(df['PTYPE'], df['LMP_CW'])
    
    # Calculate percentages for each PTYPE
    labour_market_prospects_pct = labour_market_prospects_by_ptype.div(
        labour_market_prospects_by_ptype.sum(axis=1), axis=0) * 100
    
    labour_market_prospects_by_ptype.plot(kind='bar', stacked=True, ax=ax1, 
                                         color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'])
    ax1.set_title('Labour Market Prospects Distribution by Program Type (LMP_CW)')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Count') 

    # Prevent x-axis labels from tilting
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Add count and percentage labels on the bars
    for i, c in enumerate(ax1.containers):
        # Get the percentages
        percentages = labour_market_prospects_pct.iloc[:, i].values
        
        # Create labels with both count and percentage
        labels = [f'{pct:.1f}%' for pct in percentages]
        ax1.bar_label(c, labels=labels, label_type='center')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/ptype_labour_market_prospects.png', dpi=300, bbox_inches='tight')
    
def plot_ptype_nationality(df):
    """Create a plot of the nationality distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # bar plot of nationality by ptype
    nationality_by_ptype = pd.crosstab(df['PTYPE'], df['NATION'])
    
    # Calculate percentages for each PTYPE
    nationality_pct = nationality_by_ptype.div(
        nationality_by_ptype.sum(axis=1), axis=0) * 100
    
    nationality_by_ptype.plot(kind='bar', stacked=True, ax=ax1, 
                              color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6'])
    ax1.set_title('Nationality Distribution by Program Type (NATION)')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Count') 
    ax1.legend(['1: Local', '2: Other European', '3: Asian', '4: African', '5: American'], 
               title='Nationality')

    # Prevent x-axis labels from tilting
    plt.setp(ax1.get_xticklabels(), rotation=0)

    # Add count and percentage labels on the bars
    for i, c in enumerate(ax1.containers):
        # Get the percentages
        percentages = nationality_pct.iloc[:, i].values
        
        # Create labels with percentage
        labels = [f'{pct:.1f}%' for pct in percentages]
        ax1.bar_label(c, labels=labels, label_type='center')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/ptype_nationality.png', dpi=300, bbox_inches='tight')

def plot_ptype_unemployment_rate_last_occupation(df):
    """Create a plot of the unemployment rate in the last occupation by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # box plot of unemployment rate in the last occupation by ptype
    sns.boxplot(x='PTYPE', y='PROF_AL', data=df, ax=ax1, width=0.5)

    # Set the title and labels
    ax1.set_title('Unemployment Rate in the Last Occupation by Program Type (PROF_AL)')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Unemployment Rate in the Last Occupation (PROF_AL)')

    # Add statistical annotations
    add_statistical_annotations(ax1, df, 'PTYPE', 'PROF_AL')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/ptype_unemployment_rate_last_occupation.png', dpi=300, bbox_inches='tight')

def plot_ptype_region(df):
    """Create a plot of the region distribution by program type"""
    # Create a figure
    fig = plt.figure(figsize=(20, 8))  # Wider figure to accommodate many regions
    ax1 = fig.add_subplot(111)

    # bar plot with region on x-axis and count of different ptypes
    region_by_ptype = pd.crosstab(df['REGION'], df['PTYPE'])
    region_by_ptype.plot(kind='bar', ax=ax1, stacked=True,
                         color=['#2ecc71', '#3498db', '#f39c12'], width=0.9)
    ax1.set_title('Program Type Distribution by Region')
    ax1.set_xlabel('Region (REGION)')
    ax1.set_ylabel('Count')
    ax1.legend(title='Program Type (PTYPE)')

    # Set x-axis labels to be vertical to accommodate many regions
    plt.setp(ax1.get_xticklabels(), rotation=90)

    # Add grid lines for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/ptype_region.png', dpi=300, bbox_inches='tight')

def plot_num_month_unemployed_by_ptype(df):
    """Create a plot of the number of months employed by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # box plot of num_month_employed by ptype
    sns.boxplot(x='PTYPE', y='UNEM_X0', data=df, ax=ax1, width=0.5)

    # Set the title and labels
    ax1.set_title('Number of Months Employed by Program Type (UNEM_X0)')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Number of Months Employed (UNEM_X0)')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/num_month_unemployed_by_ptype.png', dpi=300, bbox_inches='tight')

def plot_num_month_out_of_labour_force_by_ptype(df):
    """Create a plot of the number of months out of labour force by program type"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # box plot of num_month_out_of_labour_force by ptype
    sns.boxplot(x='PTYPE', y='OLF_X0', data=df, ax=ax1, width=0.5)

    # Set the title and labels
    ax1.set_title('Number of Months Out of Labour Force by Program Type (OLF_X0)')
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Number of Months Out of Labour Force (OLF_X0)')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/num_month_out_of_labour_force_by_ptype.png', dpi=300, bbox_inches='tight')
