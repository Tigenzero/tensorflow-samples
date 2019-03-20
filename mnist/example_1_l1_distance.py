"""
Calculating the L1 Distance in Tensorflow
Game Plan:
1. import mnist data (using fancy mnist object)
2. get training and test images/labels from mnist
3. create placeholders for processing images (test and training)
4. calculate distance between training digits placeholder and testing digit placeholder (more like a shell, actual calc comes later)
5. Sums all of the elements for a testing digit placeholder to one distance number per training digit
6. predicts the value (0-9) that the testing digit is based on which training digit it was closest to (arg_min)

Execution (ln 49):
1. accuracy is set to a 0 float.
2. a tensorflow session is created
3. global variables are initialized (important)
4. iterate through the test batch:
5. run the prediction variable with the feed dictionary holding the training digits and test digits
5a. because the prediction (pred) depends on distance,
        which depends on l1_distance, which depends on the training digits placeholder (training_digits_pl) and
        test digit placeholder (test_digit_pl), all will be executed to obtain the prediction
6. obtain the training prediction label (what we thought it was) and the test digit's actual label (what it is)
7. print out our results
8. obtain our accuracy and (after the loop) show our end results.
"""


import numpy
import tensorflow

from mnist.mnist_util import MNISTObject

MNIST_DATA_LOC = "mnist_data/"

# Training/Testing batch sizes. There are 60K images but the smaller the batch, the faster our process will run.
# 15000 + 1200 batch size yields higher accuracy but will take longer to process.
TRAINING_BATCH_SIZE = 5000
TESTING_BATCH_SIZE = 200

mnist = MNISTObject(MNIST_DATA_LOC, True)

training_digits, training_labels = mnist.get_training_batch(TRAINING_BATCH_SIZE)
test_digits, test_labels = mnist.get_test_batch(TESTING_BATCH_SIZE)

# We know the size of the images, so we specify the pixel size.
# However, the 'None' in this instance represents how many images we will be processing. This is currently unknown.
training_digits_pl = tensorflow.placeholder("float", [None, mnist.get_pixels()])

test_digit_pl = tensorflow.placeholder("float", [mnist.get_pixels()])

# Nearest Neighbor calculation using L1 distance
# (absolute value of the sum of the training digit location - the test digit location)
l1_distance = tensorflow.abs(tensorflow.add(training_digits_pl, tensorflow.negative(test_digit_pl)))

# Sums all the elements, getting the absolute distance
distance = tensorflow.reduce_sum(l1_distance, axis=1)

# predicts the nearest neighbor(the digit that it is closest to)
pred = tensorflow.arg_min(distance, 0)

accuracy = 0.

# Initializing the variables
init = tensorflow.global_variables_initializer()

with tensorflow.Session() as sess:
    sess.run(init)

    for i in range(len(test_digits)):
        # get nearest neighbor
        nn_index = sess.run(pred, feed_dict={training_digits_pl: training_digits, test_digit_pl: test_digits[i, :]})
        training_prediction_label = numpy.argmax(training_labels[nn_index])
        actual_label = numpy.argmax(test_labels[i])
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", training_prediction_label,
              "True Label:", actual_label)

        if training_prediction_label == actual_label:
            accuracy += 1./len(test_digits)

    print("Done!")
    print("Accuracy", accuracy)
