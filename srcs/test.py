import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style='darkgrid')

data = pd.read_csv('data.csv')
data['km'] = pd.to_numeric(data['km'], errors='coerce')
data['price'] = pd.to_numeric(data['price'], errors='coerce')

x = data['km'].values
y = data['price'].values

def slope(x, y):
    if len(x) != len(y):
        raise ValueError("len(x) is not equal to len(y)")
        
    x, y, n = np.array(x), np.array(y), len(x)
    nΣxy = n * np.sum(x * y)
    ΣxΣy = x.sum() * y.sum()
    nΣx_exp2 = n * np.sum(x ** 2)
    Σx_exp2 = x.sum() ** 2
    m = (nΣxy - ΣxΣy) / (nΣx_exp2 - Σx_exp2)
    return np.round(m, 4)

def y_intercept(x, y):
    if len(x) != len(y):
        raise ValueError("len(x) is not equal to len(y)")
    x, y, n, m = np.array(x), np.array(y), len(x), slope(x, y)
    Σy = y.sum()
    mΣx = m * x.sum()
    b = (Σy - mΣx) / n
    return np.round(b, 4)

m, b = slope(x, y), y_intercept(x, y)

print('slope:', m)
print('y intercept:', b)

data['predicted'] = data['km'].map(lambda x: m * x + b)

sns.scatterplot(x='km', y='price', data=data)
sns.lineplot(x='km', y='predicted', color='orange', lw=4, data=data)

plt.show()