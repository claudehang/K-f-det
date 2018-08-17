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
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard

import keras.backend.tensorflow_backend as K

from PIL import ImageFile
from PIL import Image

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

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(img_width, img_height))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths, ncols = 80)]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = preprocess_input( paths_to_tensor(train_files) )
valid_tensors = preprocess_input( paths_to_tensor(valid_files) )
test_tensors  = preprocess_input( paths_to_tensor(test_files) )

VGG16_model       = applications.VGG16(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
InceptionV3_model = applications.InceptionV3(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
Xception_model    = applications.Xception(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
ResNet50_model    = applications.ResNet50(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)
VGG19_model       = applications.VGG19(input_shape=(img_width, img_height, 3), weights = "imagenet", include_top=False)

base_model = VGG19_model

base_model.summary()

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
#for layer in base_model.layers:
#    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

tbCallback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# Train the model
checkpointer = ModelCheckpoint(filepath='firemodel.weights.best.hdf5', verbose=3, save_best_only=True)
hist = model.fit(train_tensors, train_targets, batch_size=32, epochs=5, 
        validation_data=(valid_tensors, valid_targets), callbacks=[checkpointer, tbCallback],
        verbose=2)
# summarize history for accuracy
plt.figure(figsize=(8, 5), dpi=100)
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
# plt.show()
plt.savefig('training_acc.png', dpi=300)


# summarize history for loss
plt.figure(figsize=(8, 5), dpi=100)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper left')
# plt.show()
plt.savefig('training_loss.png', dpi=300)


# load the model weights with the best validation loss
model.load_weights('firemodel.weights.best.hdf5')
predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_tensors]
test_accuracy = 100 * np.sum(np.array(predictions) == np.argmax(test_targets, axis=1)) / len(predictions)
print('Test accuracy: %4f%%' % test_accuracy)


