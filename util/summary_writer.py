import tensorflow


# Class intended to eliminate duplication of code required to create tensorboard summary reports
# Currently Supports images and graphs
class SummaryBuilder(object):

    def __init__(self, session, summary_path):
        self.session = session
        self.summary_writer = tensorflow.summary.FileWriter(summary_path, graph=session.graph)

    def add_image(self, image_name, image_tensor):
        summary_str = self.session.run(tensorflow.summary.image(image_name, image_tensor))
        self.summary_writer.add_summary(summary_str)

    def close_summary(self):
        self.summary_writer.close()
