
import numpy as np
import random
import matplotlib.pyplot as plt

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


# we create 20 points
np.random.seed(440)

N = 100
X = np.r_[np.random.randn(N, 2) + [5, 5], np.random.randn(N, 2) + [1,1]]
y = np.asarray( [1] * N + [-1] * N )
w_init = np.random.randn(2, 1)


def eval(X, y, w):
    y_predict = np.matmul(X, w)
    loss = 0.0
    for i in range(0, y.shape[0]):
        if y_predict[i,0] * y[i] > 0:
            loss += 0
        else:
            loss += 1

    return loss


# cost:
# if Xw * y >= 0, cost = 0
# if Xw * y < 0, cost = - Xw*y
def train(X, y, w, lr):
    n, d = X.shape
    epochs = 10000

    for t in range(epochs):
        # randomly sample a number
        k = random.randint(0, n-1)
        x = X[k, :]
        y_predict = np.matmul( x, w )

        if y_predict * y[k] <= 0:
            w = w + lr * x.reshape((2,1)) * y[k]

        print('loss', eval(X, y, w) )

    return w



lr = 0.001

learned_w = train(X, y, w_init, lr)

show_clf(X, y, w_init, learned_w)

print(w_init, learned_w)

