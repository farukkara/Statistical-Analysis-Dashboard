import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def process_data(data):
    # Data cleaning and preprocessing
    data = data.dropna()
    data = data[data['column1'] != 0]
    data['column2'] = data['column2'].apply(lambda x: x / 1000)
    
    # Feature engineering
    data['column3_log'] = np.log(data['column3'])
    data['column4_sqrt'] = np.sqrt(data['column4'])
    data['column6_norm'] = stats.zscore(data['column6'])
    
    # Data aggregation
    processed_data = data.groupby(['column1', 'column5']).agg({'column2': 'mean', 'column3': 'sum', 
                                                                'column3_log': 'std', 'column4_sqrt': 'var',
                                                                'column6_norm': 'skew'})
    processed_data = processed_data.reset_index()
    return processed_data

def plot_statistics(processed_data):
    # Boxplot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x='column5', y='column2', data=processed_data, ax=ax)
    ax.set_title('Column2 Distribution by Column5')
    plt.show()
    
    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x='column3_log', y='column4_sqrt', data=processed_data, hue='column5')
    ax.set_xlabel('Column3 Log')
    ax.set_ylabel('Column4 Sqrt')
    ax.set_title('Column3 Log vs Column4 Sqrt')
    plt.show()
    
    # Violin plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(x='column5', y='column2', data=processed_data, ax=ax)
    ax.set_title('Column2 Distribution by Column5')
    plt.show()
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='column1', y='column3', data=processed_data, hue='column5')
    ax.set_xlabel('Column1')
    ax.set_ylabel('Column3 Sum')
    ax.set_title('Column3 Sum by Column1 and Column5')
    plt.show()
    
    # Line plot
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='column1', y='column6_norm', data=processed_data, hue='column5')
    ax.set_xlabel('Column1')
   
    # Continue
    ax.set_ylabel('Column6 Normalized')
    ax.set_title('Column6 Normalized by Column1 and Column5')
    plt.show()
    
    # Pairplot
    sns.pairplot(processed_data, hue='column5')
    plt.show()
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(processed_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    plt.show()
    
    # Statistical tests
    print('ANOVA Test Results:')
    for col in ['column2', 'column3', 'column3_log', 'column4_sqrt', 'column6_norm']:
        f_val, p_val = stats.f_oneway(processed_data[processed_data['column5'] == 'group1'][col],
                                      processed_data[processed_data['column5'] == 'group2'][col],
                                      processed_data[processed_data['column5'] == 'group3'][col])
        print(f'Column: {col} - F Value: {f_val:.2f} - P Value: {p_val:.5f}')
        
    print('\nT-Test Results:')
    for col in ['column2', 'column3', 'column3_log', 'column4_sqrt', 'column6_norm']:
        t_stat, p_val = stats.ttest_ind(processed_data[processed_data['column5'] == 'group1'][col],
                                        processed_data[processed_data['column5'] == 'group2'][col])
        print(f'Column: {col} - T Statistic: {t_stat:.2f} - P Value: {p_val:.5f}')

if __name__ == '__main__':
    # Load the data into a pandas DataFrame
    data = pd.read_csv('data.csv')
    processed_data = process_data(data)
    plot_statistics(processed_data)
