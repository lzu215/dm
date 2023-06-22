import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

# load data
cancer_data = pd.read_csv('../data/Cancer_Data.csv', sep=',', index_col=0).iloc[:, :-1]
cancer_data = cancer_data.replace(to_replace={'M': 1, 'B': 0})
X, y = cancer_data.iloc[:, 1:], cancer_data.iloc[:, 0]

# standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# feature selection
lasso = LassoCV(cv=5, max_iter=10000)
sfm = SelectFromModel(lasso)
sfm.fit(X, y)
X_selected = sfm.transform(X)

# preserve data
y.reset_index(inplace=True, drop=True)
cancer_data_selected = pd.DataFrame(X_selected)
cancer_data_selected.insert(0, 'Class', y)
print(cancer_data_selected)
cancer_data_selected.to_csv('../data/Cancer_Data_Cleaned.csv', index=False)
