# -*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load in the data
data = pd.read_csv("data/moore.csv", header=None).values
X = data[:, 0].reshape(-1, 1)
Y = data[:, 1]

# plot the data
plt.scatter(X, Y)
plt.show()

# take the log to the data to make it linear
Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

# center the X data so that the values are not too large
# could do scaling, but data needs to be reversed back to original values later.
X = X - X.mean()

# build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(1,)),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9), loss="mse")


# learning rate scheduler
def schedule(epoch, lr):
    if epoch >= 50:
        return 0.0001
    return 0.001


scheduler = tf.keras.callbacks.LearningRateScheduler(schedule)

# train the model
r = model.fit(X, Y, epochs=200, callbacks=[scheduler])

# plot the loss
plt.plot(r.history["loss"], label="loss")
plt.show()

# get the slope of the line
print(model.layers)
print(model.layers[0].get_weights())            # first element is weight, second one is bias
a = model.layers[0].get_weights()[0][0, 0]      # slope

print("Time to double:", np.log(2) / a)

# Make prediction
Yhat = model.predict(X).flatten()
plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# manual calculation

# get weights
w, b = model.layers[0].get_weights()
X = X.reshape(-1, 1)
Yhat2 = (X.dot(w) + b).flatten()
np.allclose(Yhat, Yhat2)
