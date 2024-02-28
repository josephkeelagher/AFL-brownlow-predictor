import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("AFL-2022-totals.csv")

# Set all missing values to 0
df = df.fillna(0) 

# Seperate data into training and test sets.
class_values = df['BR']
feature_values = df[['UP', 'TK', 'KI', 'IF', 'HB', 'GA', 'FF', 'DI', 'CP', 'CL', 'CG']]
x_train, x_test, y_train, y_test = train_test_split(feature_values, class_values, test_size=0.1, random_state = 363) 

# Train model on training split.
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Assess Accuracy of model using k-folds cross validation.
scores = cross_val_score(linear_model, x_train, y_train, scoring='r2', cv=10)
average_score = scores.mean()
print(average_score)

# Plot results onto a scatter graph
plt.scatter(x=x_test.index, y=linear_model.predict(x_test), color='blue', label='Predicted Votes')
plt.scatter(x=y_test.index, y=y_test, color='red', label = 'Real Votes')

plt.xlabel("Player Index")
plt.ylabel("Brownlow Votes")

plt.legend()
plt.title("Predicted Brownlow votes and Real Browlow Votes Scatter Plot")
plt.savefig("testfile")

# Get accuracy of final test.
print("R²: {}".format(linear_model.score(x_test, y_test)))
