import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns

data=pd.read_csv(r'C:\Users\HP\Downloads\carprice.csv')
data.columns=data.columns.str.strip()
num_columns=data.select_dtypes(include=[np.number]).columns.tolist()

for col in num_columns:
    col_data=data[col]
    mean=col_data.mean()
    median=col_data.median()
    mode=col_data.mode()[0]
    std=col_data.std()
    percentile=np.percentile(col_data,[25,50,75])

    print(f'\n---{col}---')
    print('mean=',mean)
    print('median=',median)
    print('mode=',mode)
    print('std=',std)

    print('25th percentile=',percentile[0])
    print('50th percentile=',percentile[1])
    print('75th percentile=',percentile[2])

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(col_data,kde=True,color='salmon')
    plt.title(f'histogram with kde-{col}')
    plt.show()
    
    plt.subplot(1,2,1)
    sns.boxplot(x=col_data,color='green')
    plt.title(f'boxplot-{col}')
    plt.show()
