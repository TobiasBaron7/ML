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
import random

# PATH TO FOLDER CONTAINING DATA
_PATH_DATA  = 'C:/Users/FinalFred/Documents/SourceTree-Projects/ML/FaceDetection_FaceRecognition/data/yaleB' \
              '/ExtendedYaleB_jpg '

# DATABASE SETTINGS
_DB_NAME    = 'DB_FACES.db'
_DB_PATH    = 'C:/Users/FinalFred/Documents/SourceTree-Projects/ML/FaceDetection_FaceRecognition/data'


# GLOBAL VARS
_conn       = None      # database connection
_cursor     = None      # database cursor
_test_data  = list()    # list of paths to the test-images

# SQL STATEMENTS
# CREATE TABLES
_sql_create_table_person    = 'CREATE TABLE IF NOT EXISTS DB_Person(' \
                              'id integer PRIMARY KEY,' \
                              'SubjectNumber integer NOT NULL UNIQUE)'
_sql_create_table_featureV  = 'CREATE TABLE IF NOT EXISTS DB_FeatureVector(' \
                              'id integer PRIMARY KEY,' \
                              'PersonId integer NOT NULL,' \
                              'FeatureVector numpy_array,' \
                              'L2Norm double,' \
                              'Pose integer,' \
                              'Lightning varchar[16],' \
                              'FOREIGN KEY (PersonId) REFERENCES DB_Person(id))'
_sql_create_table_meanV     = 'CREATE TABLE IF NOT EXISTS DB_MeanVector(' \
                              'id integer PRIMARY KEY,' \
                              'PersonId integer NOT NULL,' \
                              'MeanVector numpy_array,' \
                              'FOREIGN KEY (PersonId) REFERENCES DB_Person(id))'
_sql_create_table_medianV   = 'CREATE TABLE IF NOT EXISTS DB_MedianVector(' \
                              'id integer PRIMARY KEY,' \
                              'PersonId integer NOT NULL,' \
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
def adapt_array(arr):
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
def convert_array(t):
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
        sqlite3.register_adapter(np.ndarray, adapt_array)
        sqlite3.register_converter('numpy_array', convert_array)
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


if __name__ == '__main__':
    _connect_database()
    _create_tables()

    v = np.zeros(10, dtype=np.float32)
    l2 = 0
    id = 0
    for i in range(10):
        for j in range(len(v)):
            v[j] = random.randrange(0, 10)
            l2 += v[j]
        l2 /= 10
        _cursor.execute(_sql_insert_DB_Person, [i])
        id = _cursor.lastrowid
        _cursor.execute(_sql_insert_DB_FeatureV_min, [id, v, l2])
        _cursor.execute(_sql_insert_DB_MeanV, [id, random.randrange(1, 30)])
        _cursor.execute(_sql_insert_DB_MedianV, [id, random.randrange(30, 50)])
        l2 = 0
        _conn.commit()

    """
    cur = _conn.cursor()
    cur.execute('select FeatureVector from DB_FeatureVector')
    v = cur.fetchone()[0]
    print(type(v))
    print(v)
    _create_table(_sql_create_table_person)
    _create_table(_sql_create_table_featureV)
    v = np.zeros(10, dtype=np.float32)
    for i in range(len(v)):
        v[i] = random.randrange(0, 10)
    print(v)
    _update_cursor()
    _cursor.execute('insert into DB_Person (SubjectNumber)  values(?)', '1')
    id = _cursor.lastrowid
    _conn.cursor().execute('insert into DB_FeatureVector(PersonId, FeatureVector) values (?,?)',
                           [id, v])
    _conn.commit()
    """
    _conn.close()
