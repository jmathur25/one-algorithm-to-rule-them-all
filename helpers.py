import matplotlib.pyplot as plt
import numpy as np

def na_handler(df, drop=0.5):
    nas = df.isna().sum() / len(df)
    
    medians = {}
    dropped = []
    for col in nas.index:
        if nas[col] >= drop:
            df = df.drop(col, axis=1)
            dropped.append(col)
        elif nas[col] > 0:
            medians[col] = df[col].median()
            df[col] = df[col].fillna(medians[col])
    
    return df, medians, dropped

def plot_1d(x, y, f):
    x_0 = x[y==0][:,f]
    x_1 = x[y==1][:,f]
    plt.plot(x_0, np.zeros(len(x_0)), '.', alpha=0.5, color='b')
    plt.plot(x_1, np.zeros(len(x_1)) + 1, '.', alpha=0.5, color='r')
