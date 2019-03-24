# Separate different enclosed circles of data from different classes
# This file is close to simple_blobs.py except the number of layers is increased by 2 to handle the increased complexity
# Names are added to the layers, a summary is logged, an image of the model is printed to the output, and a callback
# to stop the process when the accuracy doesn't increase after 5 epochs

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib
# If you receive the following error, use the line below: "Python is not installed as a framework."
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os

# Keras Related
from keras.models import Model
# Dense connects each layer to the following layer (or output)
from keras.layers import Dense
from keras.layers import Input
# Adam performs back propagation to layers to adjust weights and biases to minimize error during training
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from keras_samples.util.plot import plot_decision_boundary, plot_data
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
SUMMARY_OUTPUT = "data/model.png"

CALLBACK = [EarlyStopping(monitor='acc', patience=5, mode='max')]

if __name__ == "__main__":
    # Generate Data Blobs
    x, y = make_circles(n_samples=1000, factor=0.6, noise=0.1, random_state=42)

    plot = plot_data(plt, x, y)
    print("Generating Plot Window. Please look for and close the plot window")
    plot.show()

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Input Layer
    inputs = Input(shape=(2,))
    # Hidden layers
    h1 = Dense(4, activation='tanh', name="Hidden-1")(inputs)
    h2 = Dense(4, activation='tanh', name="Hidden-2")(h1)
    # Output Layer
    output = Dense(1,  activation="sigmoid", name="Output-layer")(h2)
    model = Model(inputs=inputs, outputs=output)
    # display summary
    model.summary()
    # compile the model. Minimize cross-entropy for a binary. Maximize for accuracy
    model.compile(Adam(lr=0.05), "binary_crossentropy", metrics=['accuracy'])

    # WARNING: Install graphviz in order to uncomment plot_model()
    # plot_model(model, to_file=SUMMARY_OUTPUT, show_shapes=True, show_layer_names=True)

    # Fit the model with the data from make_blobs. Make 100 cycles thorugh the data
    model.fit(x_train, y_train, epochs=100, verbose=1, callbacks=CALLBACK)
    # get loss and accuracy on test data
    eval_result = model.evaluate(x_test, y_test)
    # print test accuracy
    print("\n\nTest Loss: {} Test accuracy: {}".format(eval_result[0], eval_result[1]))
    plot_decision_boundary(model, plt, x, y).show()
