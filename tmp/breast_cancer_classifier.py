# -*- coding: utf-8 -*-
import tensorflow as tf
print(tf.__version__)

# Load in the data
from sklearn.datasets import load_breast_cancer

# Load the data
data = load_breast_cancer()

print(type(data))
print(data.keys())
print(data.data.shape)
print(data.target)
print(data.target_names)
print(data.target.shape)
print(data.feature_names)

# split the data indo traion and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)
N, D = X_train.shape

# scale the data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(D,)),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train the model
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# evaluate the model
print("train score:", model.evaluate(X_train, y_train))
print("Test score", model.evaluate(X_test, y_test))

# plot the loss
import matplotlib.pyplot as plt
plt.plot(r.history["loss"], label="loss")
plt.plot(r.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

# plot the accuracy
plt.plot(r.history["accuracy"], label="acc")
plt.plot(r.history["val_accuracy"], label="val_acc")
plt.legend()
plt.show()

# Make prediction
P = model.predict(X_test)

# round to get the actual prediction
import numpy as np
P = np.round(P).flatten()
print(P)