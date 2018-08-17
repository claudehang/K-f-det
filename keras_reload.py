# -*- coding: utf-8 -*-

# MLP for json, hdf5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

import os

import numpy as np

import urllib

url = "http://ftp.ics.uci.edu/pub/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
raw_data = urllib.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")

X = dataset[:, 0:8]
y = dataset[:, 8]

seed = 42
np.random.seed(seed)

# Create model
model = Sequential()
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, nb_epoch=150, batch_size=10, verbose=0)

# Evaluate the model
scores = model.evaluate(X, y)
print("{0}: {1:.2f}%".format(model.metrics_names[1], scores[1]*100))

# Here is the Point
# save model: JSON
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)

# save weights: HDF5
model.save_weights("model.h5")
print("Save model to disk")

# later when you want to use the model
# load json and creat model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weight into new model
loaded_model.load_weights('model.h5')
print("Loaded model from disk, OK")

# Evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, y, verbose=0)

print("{0}: {1:.2f}%".format(loaded_model.metrics_names[1], scores[1]*100))
