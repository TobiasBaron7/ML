"""
This module is responsible for  creating a database
and adding training data to it. Training data is acquired from the yale B face-data.

Idea:
Add random 70% of each subject to the database.
Therefore extract the face, compute the feature-vector and save it to the database.
Also save the pose-information to the database.
Save a list of the 30% which weren't taken, to use them later as test-data. (save path)

When done, compute meta tables as well (mean-vector of each subject, median-vector..)
"""
import io
import numpy as np
import sqlite3
import os

# PATH TO FOLDER CONTAINING DATA
_PATH_DATA = 'C:/Users/FinalFred/Documents/SourceTree-Projects/ML/FaceDetection_FaceRecognition/data/yaleB' \
             '/ExtendedYaleB_jpg'

# DATABASE SETTINGS
_DB_NAME = 'DB_FACES.db'
_DB_PATH = 'C:/Users/FinalFred/Documents/SourceTree-Projects/ML/FaceDetection_FaceRecognition/data'

# GLOBAL VARS
_conn = None  # database connection
_cursor = None  # database cursor

# SQL STATEMENTS
# CREATE TABLES
_sql_create_table_person    = 'CREATE TABLE IF NOT EXISTS DB_Person(' \
                                'id integer PRIMARY KEY,' \
                                'SubjectNumber integer NOT NULL UNIQUE)'
_sql_create_table_featureV  = 'CREATE TABLE IF NOT EXISTS DB_FeatureVector(' \
                                'id INTEGER PRIMARY KEY,' \
                                'PersonId INTEGER NOT NULL,' \
                                'FeatureVector numpy_array,' \
                                'L2Norm REAL,' \
                                'Pose INTEGER,' \
                                'Lightning TEXT,' \
                                'FOREIGN KEY (PersonId) REFERENCES DB_Person(id))'
_sql_create_table_meanV     = 'CREATE TABLE IF NOT EXISTS DB_MeanVector(' \
                                'id INTEGER PRIMARY KEY,' \
                                'PersonId INTEGER NOT NULL,' \
                                'MeanVector numpy_array,' \
                                'FOREIGN KEY (PersonId) REFERENCES DB_Person(id))'
_sql_create_table_medianV   = 'CREATE TABLE IF NOT EXISTS DB_MedianVector(' \
                                'id INTEGER PRIMARY KEY,' \
                                'PersonId INTEGER NOT NULL,' \
                                'MedianVector numpy_array,' \
                                'FOREIGN KEY (PersonId) REFERENCES DB_Person(id))'

_tables                     = [_sql_create_table_person, _sql_create_table_featureV,
                               _sql_create_table_meanV, _sql_create_table_medianV]

# INSERT STATEMENTS
_sql_insert_DB_Person       = 'INSERT INTO DB_Person (SubjectNumber)  VALUES(?)'
_sql_insert_DB_FeatureV     = 'INSERT INTO DB_FeatureVector (PersonId, FeatureVector, L2Norm, Pose, Lightning)' \
                                'VALUES(?, ?, ?, ?, ?)'
_sql_insert_DB_FeatureV_min = 'INSERT INTO DB_FeatureVector (PersonId, FeatureVector, L2Norm) VALUES(?, ?, ?)'
_sql_insert_DB_MeanV        = 'INSERT INTO DB_MeanVector (PersonId, MeanVector) VALUES (?, ?)'
_sql_insert_DB_MedianV      = 'INSERT INTO DB_MedianVector (PersonId, MedianVector) VALUES (?, ?)'

# SELECT STATEMENTS

_sql_select_all_FeatureV    = 'SELECT * FROM DB_FeatureVector'
_sql_select_person          = 'SELECT * FROM DB_PERSON WHERE id IS ?'


# http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
def _adapt_array(arr):
    """
    Converts numpy array to text.
    :param arr: numpy array
    :return:    text
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


# # http://stackoverflow.com/a/31312102/190597 (unutbu)
def _convert_array(t):
    """
    Converts text to numpy array
    :param t:   text
    :return:    numpy array
    """
    out = io.BytesIO(t)
    out.seek(0)
    return np.load(out)


def _connect_database():
    """
    Creates database at specified path and sets cursor.
    """
    global _conn, _cursor
    try:
        _conn = sqlite3.connect(_DB_PATH + '/' + _DB_NAME, detect_types=sqlite3.PARSE_DECLTYPES)
        _cursor = _conn.cursor()
        sqlite3.register_adapter(np.ndarray, _adapt_array)
        sqlite3.register_converter('numpy_array', _convert_array)
    except:
        raise


def _create_tables():
    """
    Creates tables.
    """
    global _conn, _cursor, _tables
    for table in _tables:
        try:
            _cursor = _conn.cursor()
            _cursor.execute(table)
        except:
            raise


def get_ready(create_tables=False):
    _connect_database()
    if create_tables:
        _create_tables()


# INSERTS

def insert_DB_Person(SubjectNumber):
    global _conn, _cursor, _sql_insert_DB_Person
    _cursor.execute(_sql_insert_DB_Person, [SubjectNumber])
    _conn.commit()
    return _cursor.lastrowid


def insert_DB_FeatureVector(PersonId, FeatureVector, Pose=None, Lightning=None):
    global _conn, _cursor, _sql_insert_DB_FeatureV, _sql_insert_DB_FeatureV_min
    # using min-version
    if not Pose and not Lightning:
        _cursor.execute(_sql_insert_DB_FeatureV_min, [PersonId, FeatureVector, float(np.linalg.norm(FeatureVector))])
        _conn.commit()
    # using full version with all attributes
    else:
        _cursor.execute(_sql_insert_DB_FeatureV, [PersonId, FeatureVector, float(np.linalg.norm(FeatureVector)),
                                                  Pose, Lightning])
        _conn.commit()


def insert_DB_MeanVector(PersonId, MeanVector):
    global _conn, _cursor, _sql_insert_DB_MeanV
    _cursor.execute(_sql_insert_DB_MeanV, [PersonId, MeanVector])
    _conn.commit()


def insert_DB_MedianVector(PersonId, MedianVector):
    global _conn, _cursor, _sql_insert_DB_MedianV
    _cursor.execute(_sql_insert_DB_MedianV, [PersonId, MedianVector])
    _conn.commit()


# SELECTS
def select_all_featureVectors():
    """
    Returns list of tuples with all features vectors from database.

    :return: list of tuples with feature vectors (id, PersonId, [FeatureVector], L2Norm, Pose, Lightning)
    """
    global _cursor
    _cursor.execute(_sql_select_all_FeatureV)
    return _cursor.fetchall()


def select_person(person_id):
    """
    Returns tuple with (id, SubjectNumber)

    :param person_id:   int or str with index
    :return:
    """
    global _cursor
    try:
        person_id = str(person_id)
    except:
        raise
    _cursor.execute(_sql_select_person, person_id)
    return _cursor.fetchone()


# insert training data to database
# if __name__ == '__main__':
def create_database(create_tables=False):
    import pickle
    import random
    import cv2
    import time
    import Modules.FaceExtractor as FExtractor
    import Modules.FaceFeature as FFeature
    from Modules.Enums import ImageProcessingModes as Ip
    from Modules.Enums import FaceLocalisationModes as Fl
    from Modules.Enums import FeatureExtractionModes as Fe

    # SETTINGS
    image_training_rate = 0.7   # value 0-1 determine percentage of training data

    totalImages         = 0     # all images(files) found
    totalFolders        = 0     # all subfolders found
    test_images         = []    # images used for testing

    # ImageProcessor SETTINGS
    pre_cliplimit       = 5
    pre_tile_grid_size  = (8, 8)
    face_out_size       = (224, 224)

    pre_gamma           = 3

    post_gamma          = 3

    # haarcascades SETTINGS
    scale_factor        = 1.3
    min_neighbors       = 5
    max_faces           = 1

    # META
    face_counter            = 0     # counts number of faces added to database
    second_chance_counter   = 0     # counts the times, the post_processing succeeded to find another face

    # make database ready and open connection
    get_ready(create_tables=create_tables)

    # count all files in all subdirectories
    for folder in os.listdir(_PATH_DATA):
        totalFolders += 1
        for img in os.listdir(_PATH_DATA + '/' + folder):
            totalImages += 1

    print('Total folder:', totalFolders)
    print('Total files:', totalImages)

    # iterate over all files (should be images) and extract features
    # then save them to database
    for folder in os.listdir(_PATH_DATA):
        print(totalFolders, 'left')
        totalFolders -= 1
        # Subject number extraced from folder name for yale B dataset
        # folder is like 'yaleB25', then SubjectNumber will be '25'
        SubjectNumber = folder[5:]
        # measure time for processing one folder
        time_folder = time.time()
        # list of all images in this subdirectory
        images      = os.listdir(_PATH_DATA + '/' + folder)
        img_len     = len(images)
        # random_test_list contains indexes of images which are not added to database
        random_test_list = random.sample(range(img_len), int(img_len * (1 - image_training_rate)))
        random_test_list.sort()

        # insert subject to database
        person_id = insert_DB_Person(SubjectNumber)

        # iterate every image in subdirectory and either add to db or to random_test_list
        for i in range(img_len):
            # print(totalImages, 'left')
            totalImages -= 1
            # add to random_test_list
            if len(random_test_list) > 0 and i == random_test_list[0]:
                test_images.append(_PATH_DATA + '/' + folder + '/' + images[i])
                random_test_list.pop(0)
            # add to db after feature extraction
            else:
                # read as grayscale image
                img = cv2.imread(_PATH_DATA + '/' + folder + '/' + images[i], 0)
                face_list = FExtractor.extract_faces(img,
                                                     [{Ip.CLAHE:
                                                           {'cliplimit': pre_cliplimit,
                                                            'tile_grid_size': pre_tile_grid_size}},
                                                      {Ip.GAMMA_CORRECTION:
                                                           {'gamma': post_gamma}}],
                                                     [{Fl.HAARCASCADES_FACE_PRE_TRAINED:
                                                           {'scale_factor': scale_factor,
                                                            'min_neighbors': min_neighbors,
                                                            'max_faces': max_faces}}],
                                                     face_out_size=face_out_size)
                # extract features of the face
                if face_list and len(face_list) > 0:
                    # name is 'yaleB03_P06A+035E+40.jpg' -> extract 06
                    pose = images[i][9:11]
                    # extract '035E+40'
                    lightning = images[i][12:-4]
                    face_feature = FFeature.extract_features(face_list[0], mode=Fe.CNN_VGG_16_PRE_TRAINED)

                    # save to database
                    insert_DB_FeatureVector(person_id, face_feature, pose, lightning)
                    face_counter += 1

        time_folder = time.time()-time_folder
        print('\nSTATS FOLDER:', folder)
        print('Files:\t', img_len)
        print('Total time:\t', time_folder, 's')
        print('Time/file:\t', time_folder/img_len, 's\n')
        print('Successful second attempts:', FExtractor.get_second_chance_counter())
        second_chance_counter += FExtractor.get_second_chance_counter()
        FExtractor.reset_second_chance_counter()
    print('Total faces added to DB:', face_counter)
    print('Total of successful post-processing:', second_chance_counter)
    print('Percentage of post-processing in DB:', second_chance_counter/face_counter)

    # saving test-list to file
    with open('path_test_data.txt', 'wb') as fp:
        pickle.dump(test_images, fp)

"""
READ DATA FROM FILE
with open('path_test_data.txt', 'rb') as fp:
    data = pickle.load(fp)
"""
