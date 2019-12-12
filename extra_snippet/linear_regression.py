import numpy as np
import matplotlib.pyplot as plt

# create the dataset
X = np.arange(100) + 1
Y = np.arange(100) + np.random.randn(100) * 5

# normalize the dataset
X = X / 100
Y = Y / 100

# plot the dataset
plt.scatter(X, Y)
plt.show()

# initialize the weight
w0 = 0.00000000001
w1 = 0.00000000001
cost = []


def hypothesis(X, w0, w1):
    return w0 + X * w1


def cost_function(H, Y):
    return 1/2*len(Y) * np.sum(np.square(H - Y))


def gradient_descent(H, Y, lr, w0, w1):
    w0 -= lr * 1/len(Y) * np.sum(H - Y)
    w1 -= lr * 1/len(Y) * np.sum(((H - Y) * X))
    return w0, w1


# train the model
for _ in range(100):
    H = hypothesis(X, w0, w1)
    w0, w1 = gradient_descent(H, Y, 1, w0, w1)
    cost.append(cost_function(H, Y))

plt.plot(cost)
plt.show()

plt.scatter(X, Y)
plt.plot(X, hypothesis(X, w0, w1))
plt.show()
