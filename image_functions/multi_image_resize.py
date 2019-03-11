import tensorflow
from util.file_utils import get_data_files
from util.summary_writer import SummaryBuilder

SUMMARY_PATH = "resize_image_summary"

if __name__ == "__main__":
    image_file_paths = get_data_files("data")
    print(image_file_paths)
    # create tensorflow queue
    image_queue = tensorflow.train.string_input_producer(image_file_paths)

    # read entire image file
    image_reader = tensorflow.WholeFileReader()

    with tensorflow.Session() as session:

        coordinator = tensorflow.train.Coordinator()
        threads = tensorflow.train.start_queue_runners(sess=session, coord=coordinator)

        image_dict = {}
        image_list = []
        for i in range(len(image_file_paths)):
            # Read the whole file. First param is the file name, which we don't need
            _, image_file = image_reader.read(image_queue)

            # Decode the image as a jpeg, this turns it into a Tensor used for training
            image = tensorflow.image.decode_jpeg(image_file)

            # Get Tensor of the resized image
            image = tensorflow.image.resize_images(image, [224, 224])
            image.set_shape([224, 224, 3])

            # Get an image Tensor and print its value
            image_array = session.run(image)
            print(image_array.shape)

            # If we want to show the image, we can with the line below
            # Image.fromarray(image_array.astype("uint8"), 'RGB').show()

            # OLD WAY: append image_array and add the dimension to create a 4D tensor
            # image_list.append(tensorflow.expand_dims(image_array, 0))

            # NEW WAY: tf.stack() converts a list of ran-R tensors into one rank-R+1 tensor
            image_tensor = tensorflow.stack(image_array)

            # the expand_dims adds a new dimension
            image_list.append(image_tensor)

        # close coordinator
        coordinator.request_stop()
        coordinator.join(threads)

        # Write image summary
        summary_builder = SummaryBuilder(session, SUMMARY_PATH)

        # OLD WAY: iterate through the image_list and individually add each image
        for name, tensor_object in enumerate(image_list):
            print(name)
            print(tensor_object.shape)
            summary_builder.add_image("image_{}".format(name), tensor_object)

        # NEW WAY: Stack all image tensors into a tensor containing a list of image tensors
        images_tensor = tensorflow.stack(image_list)
        summary_builder.close_summary()
