
# coding: utf-8

# In[2]:

import numpy as np
import random
import matplotlib.pyplot as plt


# In[ ]:


def show_clf(X, y, w_0, w):
    # plot the weighted data points
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.9, cmap=plt.cm.bone, edgecolor='black')

    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))
    xy = np.c_[xx.ravel(), yy.ravel()]
    Z = np.matmul(xy, w)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='b')

    Z = np.matmul(xy, w_0)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='y')

    plt.show()


# In[ ]:


# we create 20 points
np.random.seed(200)
X = np.r_[np.random.randn(10, 2) + [4, 4], np.random.randn(10, 2) + [0, 0]]
y = np.asarray( [1] * 10 + [0] * 10 )
w_init = np.random.randn(2, 1)

# In[ ]:

def eval(X, y, w):
    y_predict = np.matmul(X, w)
    loss = 0.0
    for i in range(0, y.shape[0]):
        if (y_predict[i,0]>0.5 and y[i]>0.5) or (y_predict[i,0]<=0.5 and y[i]<=0.5):
            loss += 0
        else:
            loss += 1

    return loss


# In[ ]:

def predict(x, w):
    z = np.matmul(x, w)
    y = 1.0 / (1 + np.exp(-z))
    return y

def train(X, y, w, lr):
    n, d = X.shape
    epochs = 100

    for t in range(epochs):
        # randomly sample a number
        k = random.randint(0, n-1)
        x = X[k, :]
        y_predict = predict( x, w )

        w = w + lr * x.reshape((2,1)) * y_predict * (1-y_predict) * (y[k]-y_predict)

        print('loss', eval(X, y, w) )

    return w



# In[ ]:


lr = 0.1

learned_w = train(X, y, w_init, lr)

show_clf(X, y, w_init, learned_w)

print(w_init, learned_w)


# import pdb; pdb.set_trace()
