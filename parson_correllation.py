import numpy as np
import pandas as pd

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 3, 4, 5, 6])

def pearson_correlation(x, y):
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_square = np.sum(x ** 2)
    sum_y_square = np.sum(y ** 2)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = np.sqrt((n * sum_x_square - sum_x ** 2) * (n * sum_y_square - sum_y ** 2))
    
    if denominator == 0:
        return 0
    return numerator / denominator



pearson_corr_custom = pearson_correlation(x, y)
print("Pearson Correlation Coefficient (custom formula):", pearson_corr_custom)


pearson_corr_numpy = np.corrcoef(x, y)[0, 1]
print("Pearson Correlation Coefficient (NumPy):", pearson_corr_numpy)


data = {'x': [1, 2, 3, 4, 5], 'y': [2, 3, 4, 5, 6]}
df = pd.DataFrame(data)
pearson_corr_pandas = df['x'].corr(df['y'])
print("Pearson Correlation Coefficient (pandas):", pearson_corr_pandas)
