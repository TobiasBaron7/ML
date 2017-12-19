import os
import pickle
import random
import cv2
import Modules.FaceExtractor as FExtractor
from Modules.Enums import ImageProcessingModes as Ip
from Modules.Enums import FaceLocalisationModes as Fl

# GENERAL SETTINGS
DB_PATH             = 'C:/toolkits/databases/ExtendedYaleB'
RANDOM_SEED         = 0

# SETTINGS
image_training_rate = 0.7   # value 0-1 determine percentage of training data
totalImages         = 0     # all images(files) found
totalFolders        = 0     # all subfolders found
valid_data          = []    # images where a face was detected; later splitted into train- and test-list
train_images        = []    # images used for training
test_images         = []    # images used for testing

# ImageProcessor SETTINGS
pre_cliplimit       = 5
pre_tile_grid_size  = (8, 8)
face_out_size       = (224, 224)
post_gamma          = 3

# haarcascades SETTINGS
scale_factor        = 1.3
min_neighbors       = 5
max_faces           = 1

# META
face_counter            = 0     # counts number of faces added to database
no_face_counter         = 0     # counts number of times an image is not added to the database
second_chance_counter   = 0     # counts the times, the post_processing succeeded to find another face


def count_totals(path):
    """
    Counts all files and folders in given path

    :param path: Path to Yale-Database (top-level, with identities as sub-directories)
    """
    global totalFolders, totalImages

    for f in os.listdir(path):
        totalFolders += 1
        for _ in os.listdir(path + '/' + f):
            totalImages += 1

    print('Total folder:', totalFolders)
    print('Total files:', totalImages)


def get_valid_data():
    """
    Iterates over all files in DB_PATH and tries to detect a face on every single file.
    If face was detected, the file is valid and added to the valid_data list, else the file is excluded from this list.
    """
    global DB_PATH, totalFolders, totalImages, face_counter, no_face_counter, pre_cliplimit, pre_tile_grid_size, \
        scale_factor, min_neighbors, max_faces, valid_data
    print('Validating images..')

    for folder in os.listdir(DB_PATH):
        print(totalFolders, 'folders left')
        totalFolders -= 1

        # list of all images in this subdirectory
        images = os.listdir(DB_PATH + '/' + folder)
        img_len = len(images)

        # iterate every image in subdirectory and either add to db or to random_test_list
        for i in range(img_len):
            totalImages -= 1
            # path to current image
            img_path = DB_PATH + '/' + folder + '/' + images[i]
            # read as grayscale image
            img = cv2.imread(img_path, 0)
            # extract faces from image
            face_list = FExtractor.extract_faces(img,
                                                 [{Ip.CLAHE:
                                                       {'cliplimit': pre_cliplimit,
                                                        'tile_grid_size': pre_tile_grid_size}}],
                                                 [{Fl.HAARCASCADES_FACE_PRE_TRAINED:
                                                       {'scale_factor': scale_factor,
                                                        'min_neighbors': min_neighbors,
                                                        'max_faces': max_faces}}],
                                                 face_out_size=face_out_size)

            # add image to valid_data list if face is found, else continue
            if face_list and len(face_list) > 0:
                valid_data.append(img_path)
                face_counter += 1
            else:
                no_face_counter += 1
            # print progress
            print(totalImages, 'images left', end='\r')

    save(valid_data, 'valid_data.txt')


def split_data(load_pickle=False, filename='valid_data.txt'):
    """
    Splits Yale-data in two parts:
        One for training
        One for testing
    Ratio is defined above under SETTINGS in variable 'image_training_rate', which is originally 0.7 resulting in 70%
    of data for training purpose and 30% for testing purpose.
    """
    global DB_PATH, totalFolders, totalImages, image_training_rate, RANDOM_SEED, valid_data, train_images, test_images

    # load existing valid_data-file
    if load_pickle:
        with open(filename, 'rb') as pickle_data:
            valid_data = pickle.load(pickle_data)

    # set seed to replicate sequences; originally RANDOM_SEED = 0
    random.seed = RANDOM_SEED

    # randomly generates a list with indexes for training
    training_indexes = random.sample(range(len(valid_data)), int(len(valid_data) * image_training_rate))

    for index in range(len(valid_data)):
        if index in training_indexes:
            train_images.append(valid_data[index])
        else:
            test_images.append(valid_data[index])

    save(train_images, 'training_data.txt')
    save(test_images, 'test_data.txt')


def save(file, name):
    """
    Saves given file to disk

    :param file:    File to save
    :param name:    Name of file on disk
    """
    with open(name, 'wb') as fp:
        pickle.dump(file, fp)


def print_stats():
    global valid_data, test_images, train_images

    with open('valid_data.txt', 'rb') as pickle_data:
        valid_data = pickle.load(pickle_data)
    with open('test_data.txt', 'rb') as pickle_data:
        test_images = pickle.load(pickle_data)
    with open('training_data.txt', 'rb') as pickle_data:
        train_images = pickle.load(pickle_data)

    print('Valid data total:', len(valid_data))
    print('Training data:', len(train_images))
    print('Test data:', len(test_images))


if __name__ == '__main__':

    # count all files and print them
    count_totals(DB_PATH)

    # creates a list of files, where a face has been detected and saves it
    get_valid_data()

    # randomly splits data_list into a training-set and test-set and saves them
    split_data()

    # prints length of all lists
    print_stats()
