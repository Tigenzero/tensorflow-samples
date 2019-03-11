from os import listdir
from os.path import isfile, join


def get_data_files(file_path):
    return ["{}/{}".format(file_path, f) for f in listdir(file_path) if isfile(join(file_path, f))]
