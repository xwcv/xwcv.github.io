print(__doc__)

import numpy as np
from sklearn import datasets

# import some data to play with
cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target


# shuffle
idx = np.arange(X.shape[0])
np.random.seed(0)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]


# standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

print(X.shape, y.shape)

# split train and test
X_train, y_train = X[0:499, :], y[0:499]
X_test, y_test = X[500:568, :], y[500:568]


print(X_test.shape, y_test.shape)




