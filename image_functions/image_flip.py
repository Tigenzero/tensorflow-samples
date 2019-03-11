# Purpose: To Showcase image flipping and making the results viewable in tensorboard
import matplotlib
# If you receive the following error, use the line below: "Python is not installed as a framework."
matplotlib.use('TkAgg')
import tensorflow
import matplotlib.image as mp_image
import matplotlib.pyplot as mp_pyplot


def transpose_left(image_var):
    """Rotates the image to the left by transposing it
    :param image_var: tensorflow variable - image to rotate
    :return: tensorflow transpose object - to run within the session
    """
    return tensorflow.transpose(image_var, perm=[1, 0, 2])


if __name__ == "__main__":
    # read entire image file
    image_reader = tensorflow.WholeFileReader()

    filename = "data/beer_1.jpg"
    beer_image = mp_image.imread(filename)
    print("Image shape: ", beer_image.shape)

    beer_image_var = tensorflow.Variable(beer_image, 'beer_image')

    # initializes all the tensorflow variables.
    # If this isn't created and executed, variables will not be initialized.
    init = tensorflow.global_variables_initializer()

    with tensorflow.Session() as session:
        # Necessary if any variables are called
        session.run(init)

        result = session.run(transpose_left(beer_image_var))
        mp_pyplot.imshow(result)
        mp_pyplot.show()
