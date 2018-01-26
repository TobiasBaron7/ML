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
_PATH_DATA = 'C:/Users/FinalFred/Documents/SourceTree-Projects/ML/Face_Generator/fdfr_faces/Data/Original_Yale_Train_Set'

# DATABASE SETTINGS
_DB_NAME = 'DB_FACES_56.db'
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


def get_ready():
    _connect_database()
    _create_tables()


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
    _cursor.commit(_sql_insert_DB_MeanV, [PersonId, MeanVector])
    _conn.commit()


def insert_DB_MedianVector(PersonId, MedianVector):
    global _conn, _cursor, _sql_insert_DB_MedianV
    _cursor.commit(_sql_insert_DB_MedianV, [PersonId, MedianVector])
    _conn.commit()


def system_shutdown():
    from PIL import ImageGrab
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    name = 'screenshot'
    ImageGrab.grab().save(name + '.jpeg', format='JPEG', subsampling=0, quality=100)
    os.system('shutdown -s')

if __name__ == '__main__':
    import cv2
    import time
    import ctypes
    import Modules.FaceFeature as FFeature
    from tqdm import tqdm
    from Modules.Enums import FeatureExtractionModes as Fe

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    if os.path.exists(_DB_PATH + '/' + _DB_NAME):
        is_override = input('Output-File (' + _DB_NAME + ') already exists! Override? (y/n)')
        if is_override == 'y':
            os.remove(_DB_PATH + '/' + _DB_NAME)
        else:
            new_name = input('Please enter new name for database: ')
            new_name = str(new_name)
            if len(new_name) > 0:
                _DB_NAME = new_name
            else:
                print('Invalid Name!')
                import sys
                sys.exit()

    # ------------------------

    # temporary database variables
    person_id           = 0
    subject_number      = 0

    # META
    total_images        = 0
    successful_images   = 0
    error_images        = 0

    start_time          = time.time()
    end_time            = 0
    # ----------------------

    training_data = []

    for folder in os.listdir(_PATH_DATA):
        for file in os.listdir(_PATH_DATA + '/' + folder):
            training_data.append(_PATH_DATA + '/' + folder + '/' + file)
            total_images += 1

    # connect to database
    get_ready()

    # init VGG16-Model
    FFeature.init(Fe.CNN_VGG_16_PRE_TRAINED_56)

    # iterate over training-data-list and save each file in database
    # with subject-number, pose, lightning and feature-vector
    for image_path in tqdm(training_data):
        # image_path is like 'C:/toolkits/databases/ExtendedYaleB/yaleB39/yaleB39_P08A-035E+15.pgm'
        image_info = image_path.split('/')[-1:][0][:-4]
        # Subject number is between 1 and 28
        SubjectNumber = int(image_info[5:7])
        # it's starting at 11 and number 14 is missing
        if SubjectNumber < 15:
            SubjectNumber -= 10
        else:
            SubjectNumber -= 11
        # save to database when new subject
        if SubjectNumber is not subject_number:
            subject_number = SubjectNumber
            # insert subject to database
            person_id = insert_DB_Person(subject_number)
        # name is 'yaleB03_P06A+035E+40.jpg' -> extract 06
        pose = int(image_info[9:11])
        # extract '035E+40'
        lightning = image_info[12:]

        # read as grayscale image
        try:
            img = cv2.imread(image_path, 0)
        except PermissionError or TypeError:
            error_images += 1
            continue

        # save to database
        # extract feature-vector
        face_feature = FFeature.extract_features(img)

        # save to database
        insert_DB_FeatureVector(person_id, face_feature, pose, lightning)

        successful_images += 1

    time_ = time.time() - start_time
    print('Total time:\t\t\t', time_, 's')
    print('Average time per image:\t', time_/total_images, 's')
    if error_images > 0:
        print('WARNING: ', error_images, 'IMAGES WITH ERROR!')
    else:
        print('All images successfully saved!')

    system_shutdown()
