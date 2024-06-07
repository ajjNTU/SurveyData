import pandas as pd
from scipy.stats import mode

# Sample data setup
data = pd.DataFrame({
    'Stratum': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B'],
    'Question 1': [1, 2, 2, 3, None, 3, None, 2],
    'Question 2': [None, 2, 3, None, 5, 3, 4, None]
})


# Function to impute median or mode
def impute_with_group_median_mode(column, group_column, method='median'):
    if method == 'median':
        return column.fillna(column.groupby(data[group_column]).transform('median'))
    elif method == 'mode':
        # Mode can be more than one value, ensure single value is returned
        return column.fillna(column.groupby(data[group_column]).transform(lambda x: mode(x).mode[0]))


# Apply the imputation
data['Question 1'] = impute_with_group_median_mode(data['Question 1'], 'Stratum', method='median')
data['Question 2'] = impute_with_group_median_mode(data['Question 2'], 'Stratum', method='mode')

print(data)
