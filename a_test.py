import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

def one_hot_encoding(Y):
    """
    Create one hot encoded vector from a binary class vector.
    """

    Y_one_hot = np.ones(len(Y))
    Y_one_hot[Y == 1] = 0
    Y = Y.reshape([len(Y), 1])
    Y_one_hot = Y_one_hot.reshape([len(Y_one_hot),1])
    Y = np.concatenate((Y_one_hot, Y), axis=1)
    return Y

def deep_net(max_features, num_classes, num_layers=2, layer_shrinkage=0.1, dropout=0.5):
    """
    Returns a compiled deep net model based on hyperparameters passed to the function.
    """

    model = Sequential()
    prev_layer_size = max_features
    for i in range(num_layers):
        curr_layer_size = int(layer_shrinkage*prev_layer_size)
        model.add(Dense(curr_layer_size, input_shape=(prev_layer_size,), init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
        prev_layer_size = curr_layer_size

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    print("Compiling model")
    # model.compile(loss='categorical_crossentropy', optimizer='adam')
    # Version 1.0 compatible
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


X = np.random.randint(2, size=(100, 1000))
y = np.random.randint(2, size=100)
model = deep_net(1000, 2)
history = model.fit(X, one_hot_encoding(y), nb_epoch=5, batch_size=1, verbose=1)

model.save_weights("test_weights.h5")