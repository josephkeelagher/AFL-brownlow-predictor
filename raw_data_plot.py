import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from sklearn.model_selection import train_test_split

# read csv and impute missing data
df = pd.read_csv("AFL-2022-totals.csv")
df = df.replace(np.nan,0)

my_path = '/home/raw_graphs'
isFile = os.path.isdir(my_path)

# exit code if folder already exists
if isFile == True:
    exit()

# make new folder for graphs
os.makedirs('raw_graphs')

# split into training and test data
train_df, test_df = train_test_split(df, test_size=0.1, random_state=363)

# make graphs
features = ['GM', 'KI', 'MK', 'HB', 'DI', 'GL', 'BH', 'HO', 'TK', 'RB', 'IF', 'CL', 'CG', 'FF', 'FA', 'CP', 'UP', 'CM', 'MI', '1%', 'BO', 'GA']
names = ['Games', 'Kicks', 'Marks', 'Handballs', 'Disposals', 'Goals', 'Behinds', 'Hit Outs', 'Tackles', 'Rebound 50\'s', 'Inside 50\'s', 'Clearances', 'Clangers', 
'Free Kicks for', 'Free kicks against', 'Contested possessions', 'Uncontested possessions','Contested marks', 'Marks inside 50', 'One percenters', 'Bounces', 'Goal assist']

# plot graphs and save fig to file created
for feature, name in zip(features, names):
    ax1 = train_df.plot.scatter(x=feature, y='BR')
    title = name + ' vs Brownlow votes'
    plt.title(title)

    # Fit a linear regression line
    X = train_df[feature].values.reshape(-1, 1)
    y = train_df['BR'].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Plot the linear regression line
    ax1.plot(X, y_pred, color='red')

    # save the plot as png
    my_file = feature + '_' + 'BR.png'
    full_path = os.path.join(my_path, my_file)
    plt.savefig(full_path) 
    plt.close()
