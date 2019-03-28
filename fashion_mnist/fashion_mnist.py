# Train a ConvNet on the MNIST fashion dataset
# Code is based on the MNIST example on Keras.io
# www.github.com/zalandresearch/fashion-mnist for more info as well as ideas on how to increase accuracy.



from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib
# If you receive the following error, use the line below: "Python is not installed as a framework."
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# suppress warnings and info messages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Number of classes - do not change unless data changes
num_classes = 10
batch_size = 128
epochs = 24

# image dimensions
img_rows, img_cols = 28, 28

# load data, shuffled, and shuffled and split to training and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Deal with format issues between different backends.
if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


# Type convert and scale the test and training data
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define the Model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# define compile to minimize categorical loss, use ada delta optimized, and optimize to maximize accuracy
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# evaluate the model with the test data
score = model.evaluate(x_test, y_test, verbose=0)
print('Test Loss: {}'.format(score[0]))
print('Test Accuracy: {}'.format(score[1]))

epoch_list = list(range(1, len(hist.history['acc'])+1))
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(("Training Accuracy", "Validation Accuracy"))
plt.show()

# Last Run :
# Test Loss: 0.25105614386796954
# Test Accuracy: 0.917
