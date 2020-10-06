import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load training dataset from csv
train = pd.read_csv("task0/train.csv")

# Extract dependent variable
y = train.iloc[:,1]
y = y.to_numpy()

# Extract features
X = train.iloc[:,2:]
X = X.to_numpy()

# Fit model
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_

# Make predictions on test set
test = pd.read_csv('task0/test.csv')
Ids = test.iloc[:,0]
test = test.iloc[:,1:].to_numpy()
predictions = reg.predict(test)

d = {'Id': Ids, 'y': predictions}
submission = pd.DataFrame(data=d)
submission.to_csv('task0/submission.csv', index=False)

###################################
#     SIMPLY TAKING THE MEAN      #
###################################
test = pd.read_csv('task0/test.csv')
test['mean'] = test.iloc[:, 1:].mean(axis=1)
d = {'Id': Ids, 'y': test['mean']}
submission2 = pd.DataFrame(data=d)
submission2.to_csv('task0/submission_mean.csv', index=False)
