import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
# avoid code crash when running without gui
plt.switch_backend('agg')
from tqdm import tqdm

from sklearn.datasets import load_files
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout, Flatten, Dense
from keras.models import Sequential, Model
from keras import applications
# from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

# set the optimizer
from keras.optimizers import RMSprop

import keras.backend.tensorflow_backend as K

from PIL import ImageFile
from PIL import Image


from my_gener import DataGenerator

NEED_GPU_MEM_WORKAROUND = False
if (NEED_GPU_MEM_WORKAROUND):
    print('Working around TF GPU mem issues')
    import tensorflow as tf
    import keras.backend.tensorflow_backend as ktf

    def get_session(gpu_fraction=0.6):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction,
                                allow_growth=True)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    ktf.set_session(get_session())


img_width, img_height = 224, 224 # change based on the shape/structure of your images
num_classes = 2 # Fire or Safe

# import dataset
def load_dataset(path):
    data = load_files(path)
    fire_files = np.array(data['filenames'])
    fire_targets = np_utils.to_categorical(np.array(data['target']), num_classes)
    return fire_files, fire_targets


train_files, train_targets = load_dataset('MY/train')
valid_files, valid_targets = load_dataset('MY/valid')
test_files, test_targets = load_dataset('MY/test')

print('There are %s total fire images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training fire images.' % len(train_files))
print('There are %d validation fire images.' % len(valid_files))
print('There are %d test fire images.'% len(test_files))


# =============================================== #
params = {'dim': (img_width, img_height),
          'batch_size': 32,
          'n_classes': num_classes,
          'n_channels': 3,
          'shuffle': True}

train_labels = dict(zip(train_files, train_targets[:, 1]))
training_generator = DataGenerator(train_files, train_labels, **params)

valid_labels = dict(zip(valid_files, valid_targets[:, 1]))
validation_generator = DataGenerator(valid_files, valid_labels, **params)

VGG19_model       = applications.VGG19(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)

base_model = VGG19_model

base_model.summary()

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# tbCallback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Train the model
# checkpointer = ModelCheckpoint(filepath='firemodel.weights.best.hdf5', verbose=3, save_best_only=True)

hist = model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=True, workers=6,
                           epochs=3, verbose=2)


# hist = model.fit(train_tensors, train_targets, batch_size=32, epochs=3,
#         validation_data=(valid_tensors, valid_targets), callbacks=[checkpointer, tbCallback],
#         verbose=2)



# =============================================== #






# # pre-process the data for Keras
# train_tensors = preprocess_input( paths_to_tensor(train_files) )
# print(type(train_tensors))
# print(train_tensors.shape)
#
#
# valid_tensors = preprocess_input( paths_to_tensor(valid_files) )
# test_tensors  = preprocess_input( paths_to_tensor(test_files) )
#
# # VGG16_model       = applications.VGG16(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
# # InceptionV3_model = applications.InceptionV3(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
# # Xception_model    = applications.Xception(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
# # ResNet50_model    = applications.ResNet50(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
# VGG19_model       = applications.VGG19(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
#
# base_model = VGG19_model
#
# base_model.summary()
#
# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name)
# for layer in base_model.layers:
#     layer.trainable = False
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
#
# predictions = Dense(num_classes, activation='softmax')(x)
#
# model = Model(inputs=base_model.input, outputs=predictions)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.summary()
#
# tbCallback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#
# # Train the model
# checkpointer = ModelCheckpoint(filepath='firemodel.weights.best.hdf5', verbose=3, save_best_only=True)
# # hist = model.fit(train_tensors, train_targets, batch_size=32, epochs=3,
# #         validation_data=(valid_tensors, valid_targets), callbacks=[checkpointer, tbCallback],
# #         verbose=2)
