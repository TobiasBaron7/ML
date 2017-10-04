from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend
import numpy as np


# CNN VGG 16 variables
_path_cnn_16_weights    = 'C:/VGG/vgg16_weights.h5'
_vgg16_model            = None


# TODO
def _init_vgg16_model(weights_path=None, classification=False, test=True):
    global _vgg16_model

    # testing with model using pretrained weights of imagenet
    # inclue_top=False: use as feature extraction, not classification
    if test:
        _vgg16_model = VGG16(weights='imagenet', include_top=False)
        return

    _vgg16_model = Sequential()
    _vgg16_model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    _vgg16_model.add(Conv2D(64, (3, 3), activation='relu'))
    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(64, (3, 3), activation='relu'))
    _vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(128, (3, 3), activation='relu'))
    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(128, (3, 3), activation='relu'))
    _vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(256, (3, 3), activation='relu'))
    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(256, (3, 3), activation='relu'))
    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(256, (3, 3), activation='relu'))
    _vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(512, (3, 3), activation='relu'))
    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(512, (3, 3), activation='relu'))
    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(512, (3, 3), activation='relu'))
    _vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(512, (3, 3), activation='relu'))
    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(512, (3, 3), activation='relu'))
    _vgg16_model.add(ZeroPadding2D((1, 1)))
    _vgg16_model.add(Conv2D(512, (3, 3), activation='relu'))
    _vgg16_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    _vgg16_model.add(Flatten())
    _vgg16_model.add(Dense(4096, activation='relu'))
    _vgg16_model.add(Dropout(0.5))
    _vgg16_model.add(Dense(4096, activation='relu'))
    _vgg16_model.add(Dropout(0.5))

    if classification:
        _vgg16_model.add(Dense(1000, activation='softmax'))

    if weights_path:
        _vgg16_model.load_weights(weights_path)


# TODO
def _cnn_vgg_16(img):
    global _vgg16_model

    # set dimension order according to backend used
    # 'tf' = tensorflow, 'th = theano
    if backend.backend() == 'tensorflow':
        backend.set_image_dim_ordering('tf')
    elif backend.backend() == 'theano':
        backend.set_image_dim_ordering('th')

    if not _vgg16_model:
        _init_vgg16_model()

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    features = _vgg16_model.predict(x)
    print(features.shape)
    print(type(features))
    return features


# TODO
def extract_features(img):
    pass


if __name__ == '__main__':
    img_path = '../data/test/side/side1.jpg'
    _cnn_vgg_16(img=image.load_img(img_path, target_size=(224, 224)))

