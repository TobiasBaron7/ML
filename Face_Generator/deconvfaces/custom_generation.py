#import deconvfaces.faces.generate as gen
import os
import datetime
from keras import backend as K
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import scipy

_PATH_MODEL     = 'output/100_epochs_tensorflow_FaceGen.YaleFaces.model.d5.adam.h5'
_PATH_OUTPUT    = 'test_output'

_NUM_IMAGES     = 1

# IDENTITY SPECS
_IDENTITY       = 0
_POSE           = 0
_LIGHTING       = 0


identity        = np.zeros(shape=28)
pose            = np.zeros(shape=10)
lighting        = np.zeros(shape=4)

identity[_IDENTITY] = 1
pose[_POSE] = 1
lighting[_LIGHTING] = 1

def reshape_array(a):
    t = []
    t.append(a)
    return np.array(t)


identity = reshape_array(identity)
pose = reshape_array(pose)
lighting = reshape_array(lighting)


if not os.path.exists(_PATH_OUTPUT):
    os.makedirs(_PATH_OUTPUT)
else:
    _PATH_OUTPUT = _PATH_OUTPUT + str(datetime.date.today())
    os.makedirs(_PATH_OUTPUT)

print('Loading model..')
model = load_model(_PATH_MODEL)

print('Generating images..')

count = 0
batch_size = 32

for idx in tqdm(range(0, _NUM_IMAGES)):

    batch = {
        'identity': identity,
        'pose':     pose,
        'lighting': lighting,
    }

    gen = model.predict_on_batch(batch)

    for i in range(0, gen.shape[0]):
        if K.image_dim_ordering() == 'th':
            image[:, :] = gen[i, 0, :, :]
        else:
            image = gen[i,:,:,0]
        image = np.array(255*np.clip(image,0,1), dtype=np.uint8)
        file_path = os.path.join(_PATH_OUTPUT, '{:05}.{}'.format(count, 'jpg'))
        scipy.misc.imsave(file_path, image)
        count += 1
