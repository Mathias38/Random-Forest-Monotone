"""
Created on Thu Oct  8 10:53:30 2020

@author: mathias chastan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from source.RandomForestMonotone import RandomForestClassifier
from sklearn.metrics import f1_score

'''------------------------------------------------- DATA PREP ----------------------------------------------------'''
test = pd.read_csv('data/test.csv' , sep=';')
train = pd.read_csv('data/train.csv' , sep=';')

print("TRAIN")
groups = train.groupby("Label")
for name, group in groups:
    plt.plot(group["x1"], group["x2"], marker="o", linestyle="", label=name)
plt.legend()
plt.show()
print("TEST")
groups = test.groupby("Label")
for name, group in groups:
    plt.plot(group["x1"], group["x2"], marker="o", linestyle="", label=name)
plt.legend()
plt.show()

x_test = test[['x1','x2']]

x_train_list = train.values.tolist()
x_test_list = x_test.values.tolist()
x = train.drop(['Label'], axis = 1)
y = train['Label']

'''---------------------------------------------------------------------------------------------------------------------'''
    
'''------------------------------------------------------------- MODEL -------------------------------------------------'''

rf = RandomForestClassifier(nb_trees=100, max_features = None)
rf_monotone = RandomForestClassifier(nb_trees=100, max_features = None)

rf.fit(x,y)
rf_monotone.fit_with_monotony_drop_out(x,  
                                       y,
                                       500,
                                       [0, 1],  
                                       [],
                                       [100, 100],
                                       [100, 100],
                                       [])

preds = rf.pred(x_test_list)

f1_score_rf = f1_score(test['Label'], preds)

print("f1_score rf")
print(f1_score_rf)

preds_monotone = rf_monotone.pred(x_test_list)

f1_score_rf_monotone = f1_score(test['Label'], preds_monotone)

print("f1_score rf_monotone")
print(f1_score_rf_monotone)

approved_trees_percent = rf_monotone.accepted_trees_percent

print("approved_trees_percent")
print(approved_trees_percent)

'''-----------------------------------------------------------------------------------------------------------------'''


'''---------------------------------------------------PLOT DECISIONS------------------------------------------------'''

scatter_x = np.array(x_test.iloc[:,0])
scatter_y = np.array(x_test.iloc[:,1])
group = preds_monotone
cdict = {0: 'blue', 1: 'red'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    plt.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 10)
ax.legend()
plt.title("RF MONOTONE x = " + train.columns[0] + " y = " + train.columns[1])
plt.show()

group = preds

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    plt.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 10)
ax.legend()
plt.title("RF x = " + train.columns[0] + " y = " + train.columns[1])
plt.show()
                        
group = test['Label']

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    plt.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g, s = 10)
ax.legend()
plt.title("REAL x = " + train.columns[0] + " y = " + train.columns[1])
plt.show()
'''-----------------------------------------------------------------------------------------------------------------'''   