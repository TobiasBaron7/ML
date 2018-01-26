import os
from shutil import copyfile


_PATH_ORIGIN    = 'Original_Yale_Cropped'
_OUTPUT_DIR     = 'Original_Yale_Cropped_Pose_0-5'


def create_dir_if_not_exist(path):
    """
    Checks if given directory already exists.
    If not, creates this directory.

    :param path:  Directory to check
    """
    if not os.path.exists(path):
        os.makedirs(path)


create_dir_if_not_exist(_OUTPUT_DIR)
for sub_folder in os.listdir(_PATH_ORIGIN):
    create_dir_if_not_exist(_OUTPUT_DIR + '/' + sub_folder)
    for file in os.listdir(_PATH_ORIGIN + '/' + sub_folder):
        try:
            pose = int(file[9:11])
        except ValueError:
            raise
        if pose <= 5:
            copyfile(_PATH_ORIGIN + '/' + sub_folder + '/' + file, _OUTPUT_DIR + '/' + sub_folder + '/' + file)



# copyfile(file, _OUTPUT_DIR + '/' + file_name)

