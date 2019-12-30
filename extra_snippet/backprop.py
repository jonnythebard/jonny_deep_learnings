import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

D = 2
M = 3
K = 3

X1 = np.random.randn(500, D) + np.array([0, -2])
X2 = np.random.randn(500, D) + np.array([2, 2])
X3 = np.random.randn(500, D) + np.array([-2, 2])

X = np.vstack([X1, X2, X3])
Y = np.array([0] * 500 + [1] * 500 + [2] * 500)

N = len(Y)

T = np.zeros((N, K))
for i in range(N):
    T[i, Y[i]] = 1

plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
plt.show()


def forward(X, W1, b1, W2, b2):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))   # sigmoid
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)  # softmax
    return Y, Z

def derivative_w2(Z, T, Y):
    return -Z.T.dot(T - Y)

def derivative_w1(X, Z, T, Y, W2):
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    return -X.T.dot(dZ)

def derivative_b2(T, Y):
    return -(T - Y).sum(axis=0)

def derivative_b1(T, Y, W2, Z):
    return -((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

def cost(T, Y):
    tot = -(T * np.log(Y))
    return tot.sum()

def classification_rate(Y, P):
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total


# initialize weights
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)


learning_rate = 1e-3
costs = []
for epoch in range(1000):
    output, hidden = forward(X, W1, b1, W2, b2)
    if epoch % 100 == 0:
        c = cost(T, output)
        P = np.argmax(output, axis=1)
        r = classification_rate(Y, P)
        print("cost:", c, "classification_rate:", r)
        costs.append(c)

    # gradient descent
    W2 -= learning_rate * derivative_w2(hidden, T, output)
    b2 -= learning_rate * derivative_b2(T, output)
    W1 -= learning_rate * derivative_w1(X, hidden, T, output, W2)
    b1 -= learning_rate * derivative_b1(T, output, W2, hidden)

plt.plot(costs)
plt.show()