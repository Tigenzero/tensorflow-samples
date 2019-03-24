# Separate different blobs of data from different classes

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib
# If you receive the following error, use the line below: "Python is not installed as a framework."
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
# Keras Related
from keras.models import Sequential
# Dense connects each layer to the following layer (or output)
from keras.layers import Dense
# Adam performs back propagation to layers to adjust weights and biases to minimize error during training
from keras.optimizers import Adam
from keras_samples.util.plot import plot_decision_boundary, plot_data
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


if __name__ == "__main__":
    # Generate Data Blobs
    x, y = make_blobs(n_samples=1000, centers=2, random_state=42)

    plot = plot_data(plt, x, y)
    print("Generating Plot Window. Please look for and close the plot window")
    plot.show()

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Simple Sequential Model
    model = Sequential()
    # Add a Dense fully connected layer with the 1 Neuron. Using input_shape = (2,) says the input will be arrays of the
    # form (*, 2). The first dimension will be an unspecified number of batches (rows) of data. The second dimension
    # is 2 which are the x, y positions of each data element. the sigmoid activation function is used to return 0 or 1,
    # signifying the data cluster the position is predicted to belong to.
    model.add(Dense(1, input_shape=(2,), activation="sigmoid"))
    # compile the model. Minimize cross-entropy for a binary. Maximize for accuracy
    model.compile(Adam(lr=0.05), "binary_crossentropy", metrics=['accuracy'])
    # Fit the model with the data from make_blobs. Make 100 cycles thorugh the data
    # Set verbose to 0 to supress progress messages
    model.fit(x_train, y_train, epochs=100, verbose=0)
    # get loss and accuracy on test data
    eval_result = model.evaluate(x_test, y_test)
    # print test accuracy
    print("\n\nTest Loss: {} Test accuracy: {}".format(eval_result[0], eval_result[1]))
    plot_decision_boundary(model, plt, x, y).show()
