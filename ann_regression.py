# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'   # this eliminates libiomp5.dylib error in mac

# make the dataset
N = 1000
X = np.random.random((N, 2)) * 6 - 3
Y = np.cos(2*X[:, 0]) + np.cos(3*X[:, 1]) # y = cos(2x1) + cos(3x2)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()
plt.close(fig)


# build the model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, input_shape=(2,), activation="relu"),
  tf.keras.layers.Dense(1)
])

# compile and train
opt = tf.keras.optimizers.Adam(0.01)
model.compile(optimizer=opt, loss="mse")
r = model.fit(X, Y, epochs=100)

# plot the loss
# plt.plot(r.history["loss"], label="loss")

# plot the prediction surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# surface plot
line = np.linspace(-3, 3, 50)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
Yhat = model.predict(Xgrid).flatten()
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat, linewidth=0.2, antialiased=True)
plt.show()
