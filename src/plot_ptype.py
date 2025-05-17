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

def plot_ptype_descriptive_statistics(df):
    """Create publication-quality plots for program type analysis
    """
    set_style()
    plot_sex_by_ptype(df)
    plot_age_by_ptype(df)
    plot_duration_by_ptype(df)
    plot_ptype_frequency(df)
    plot_ptype_past_income(df)

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
    gender_by_ptype.plot(kind='bar', stacked=True, ax=ax1, 
                        color=['#2ecc71', '#e74c3c'])
    ax1.set_title('Sex Distribution by Program Type', pad=20)
    ax1.set_xlabel('Program Type (PTYPE)')
    ax1.set_ylabel('Count')
    ax1.legend(['1.0', '2.0'], title='Sex')
    
    # Add count labels on the bars
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%d', label_type='center')
    
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
