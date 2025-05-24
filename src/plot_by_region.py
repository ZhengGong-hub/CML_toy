import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_by_region(df):
    """Create plots of the unemployment rate and sector shares by region"""
    plot_region_unemployment_rate(df)
    plot_region_service_sector_share(df)

def plot_region_unemployment_rate(df):
    """Create a plot of the unemployment rate by region"""
    # Create a figure
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(111)

    # bar plot of unemployment rate by region
    sns.barplot(x='REGION', y='REG_AL', data=df, ax=ax1, width=0.5)

    # xaxis rotation 90 degrees
    plt.xticks(rotation=90)

    # Set the title and labels
    ax1.set_title('Unemployment Rate by Region (REG_AL)')
    ax1.set_xlabel('Region (REGION)')
    ax1.set_ylabel('Unemployment Rate (REG_AL)')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/region_unemployment_rate.png', dpi=300, bbox_inches='tight')


def plot_region_service_sector_share(df):
    """Create a plot of the sector shares by region for service, production, and agriculture sectors"""
    # Create a figure
    fig = plt.figure(figsize=(20, 8))  # Wider figure to accommodate many regions
    ax1 = fig.add_subplot(111)

    # Define sectors to plot with their properties
    sectors = [
        {'col': 'REG_SER', 'name': 'Service', 'color': '#3498db'},
        {'col': 'REG_PRO', 'name': 'Production', 'color': '#e74c3c'},
        {'col': 'REG_AGRI', 'name': 'Agriculture', 'color': '#2ecc71'}
    ]
    
    # Create a DataFrame to hold all sector data by region
    sector_data = pd.DataFrame()
    
    # Prepare data for each sector
    for sector in sectors:
        sector_data[sector['name']] = df.groupby('REGION')[sector['col']].mean()
    
    # Plot stacked bar chart with all sectors
    sector_data.plot(kind='bar', ax=ax1, stacked=True, 
                    color=[s['color'] for s in sectors])
    
    ax1.set_title('Sector Shares by Region')
    ax1.set_xlabel('Region (REGION)')
    ax1.set_ylabel('Sector Share')
    ax1.legend(title='Sectors', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis labels to be vertical to accommodate many regions
    plt.setp(ax1.get_xticklabels(), rotation=90)
    
    # Add grid lines for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output_data/region_sector_shares.png', dpi=300, bbox_inches='tight')