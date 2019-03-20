# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


class MNISTObject(object):
    """
    MNIST data object to handle all functions necessary
    """
    # The images are 28 x 28, so this is 784 pixels
    MNIST_IMAGE_PIXELS = 784

    def __init__(self, mnist_loc, one_hot=False):
        self.mnist = input_data.read_data_sets(mnist_loc, one_hot=one_hot)

    def get_training_batch(self, batch_size):
        return self.mnist.train.next_batch(batch_size)

    def get_test_batch(self, batch_size):
        return self.mnist.test.next_batch(batch_size)

    def get_pixels(self):
        return self.MNIST_IMAGE_PIXELS
