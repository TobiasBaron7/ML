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


def identify(v):
    global _feature_vectors

    # identification
    # calc distance from v to each feature vector
    min_d = 100
    r = None
    for row in _feature_vectors:
        # if distance == 'euclidean':
            # d = euclidean(v, row[2])
        d = cosine((1/norm(v))*v, (1/row[3])*row[2])

        if d < min_d:
            min_d = d
            r = row

    return r, min_d

