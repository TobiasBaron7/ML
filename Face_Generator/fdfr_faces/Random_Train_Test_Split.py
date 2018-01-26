import os
import random
from shutil import copyfile

# GENERAL SETTINGS
_PATH_SOURCE        = 'Original_Yale_Cropped_Pose_0-5'
_PATH_TRAIN_SET     = 'Original_Yale_Train_Set_Pose_0-5'
_PATH_TEST_SET      = 'Original_Yale_Test_Set_Pose_0-5'
RANDOM_SEED         = 0

# SETTINGS
image_training_rate = 0.7   # value 0-1 determine percentage of training data
totalImages         = 0     # all images(files) found
totalFolders        = 0     # all subfolders found
valid_data          = []    # images where a face was detected; later splitted into train- and test-list
train_images        = []    # images used for training
test_images         = []    # images used for testing

# META
face_counter            = 0     # counts number of faces added to database
no_face_counter         = 0     # counts number of times an image is not added to the database
second_chance_counter   = 0     # counts the times, the post_processing succeeded to find another face

def create_dir_if_not_exist(path):
    """
    Checks if given directory already exists.
    If not, creates this directory.

    :param path:  Directory to check
    """
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    create_dir_if_not_exist(_PATH_TRAIN_SET)
    create_dir_if_not_exist(_PATH_TEST_SET)

    for folder in os.listdir(_PATH_SOURCE):
        create_dir_if_not_exist(_PATH_TRAIN_SET + '/' + folder)
        create_dir_if_not_exist(_PATH_TEST_SET + '/' + folder)

        # list of all images in this subdirectory
        images = os.listdir(_PATH_SOURCE + '/' + folder)
        img_len = len(images)

        is_training_set = [False] * img_len

        # randomly generates a list with indexes for training
        training_data = random.sample(range(len(images)), int(img_len * image_training_rate))

        for i in training_data:
            is_training_set[i] = True

        for i in range(img_len):
            if is_training_set[i]:
                copyfile(_PATH_SOURCE + '/' + folder + '/' + images[i],
                         _PATH_TRAIN_SET + '/' + folder + '/' + images[i])
            else:
                copyfile(_PATH_SOURCE + '/' + folder + '/' + images[i],
                         _PATH_TEST_SET + '/' + folder + '/' + images[i])
