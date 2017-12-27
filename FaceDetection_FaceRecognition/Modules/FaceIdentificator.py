import Modules.DB_Helper as db
from scipy.spatial.distance import euclidean, cosine
from numpy.linalg import norm

# list of all feature vectors from database
_feature_vectors = None


def get_ready():
    global _feature_vectors
    # on first run connect to database and get feature vectors
    if not _feature_vectors:
        print('connecting to database and collecting information..')
        db.get_ready(create_tables=False)
        _feature_vectors = db.select_all_featureVectors()


def identify(v, distance='cosine'):
    """
    Compares given vector to all vectors in the database and returns the closest one
    according to given distance-metric.

    :param v:           input vector
    :param distance:    distance-metric; supports: 'eudlidean', 'cosine'
    :return:            database-row and distance to this row's vector and number of comparisons;
                        if not found row is None and distance is infinity
    """
    global _feature_vectors

    if not _feature_vectors:
        get_ready()

    # identification
    # calc distance from v to each feature vector
    min_d = float('inf')
    num_comp = 0
    r = None
    if distance is 'euclidean':
        for row in _feature_vectors:
            d = euclidean(v, row[2])

            if d < min_d:
                min_d = d
                r = row
    elif distance is 'cosine':
        v = (1/norm(v))*v
        for row in _feature_vectors:
            num_comp += 1
            d = cosine(v, (1/row[3])*row[2])

            if d < min_d:
                min_d = d
                r = row

    return r, min_d, num_comp

