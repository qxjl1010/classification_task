from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

import random
# using knn
# for details, visit:
# https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#sphx-glr-auto-examples-neighbors-plot-regression-py


# you can change Nighbor value here
NeighborNumber = 3


# since the dataset has more than 280k class_0 and only 492 class_1
# we need to extract the same number of class_0 as class_1
def get_0(raw_array):
    class_0 = 0
    class_1 = 0
    balance_array = []

    for line in raw_array:
        # manually set number to 492, according to the result get by check_class.py
        if line[-1] == 0 and class_0 < 492:
            balance_array.append(line)
            class_0 += 1
        elif line[-1] == 1:
            balance_array.append(line)
            class_1 += 1
    return np.asarray(balance_array)

# read csv file
data = pd.read_csv('fraud_prep.csv')
array = data.values

# get clean dataset
array = get_0(array)

# get target and features
target = array[:,-1]
features = array[:,:-1]

# shuffle dataset
li=list(range(len(target)))
random.shuffle(li)
shuffled_features = [x for _,x in sorted(zip(li,features))]
shuffled_target = [x for _,x in sorted(zip(li,target))]

# read model
neigh = KNeighborsClassifier(n_neighbors=NeighborNumber)

# train model
neigh.fit(shuffled_features, shuffled_target) 

# you can predict here
# need a function compare predict value and real result and calculate the accuracy
print(neigh.predict([array[20][:-1]]))
