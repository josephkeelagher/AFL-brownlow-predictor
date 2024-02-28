import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

# making dataframe 
df = pd.read_csv("AFL-2022-totals.csv") 
# make NaN values 0
df = df.replace(np.nan,0) 
categories = ['GM', 'KI', 'MK', 'HB', 'DI', 'GL', 'BH', 'HO', 'TK', 'RB', 'IF', 'CL', 'CG', 'FF', 'FA', 'CP', 'UP', 'CM', 'MI', '1%', 'BO', 'GA']

# split the dataset into train and test data so feature selection can be performed on t
train_df, test_df = train_test_split(df, test_size=0.1, random_state=363)

# find pearson correlation between features and BR
PEARSON_THRESHOLD = 0.5
pearson_features = []

corr_mat = train_df.corr(numeric_only=True)

for x in categories:
    print(x, corr_mat[x]['BR'])
    if corr_mat[x]['BR'] > PEARSON_THRESHOLD:
        pearson_features.append(x)

print(sorted(pearson_features, reverse=True))

