# coding=UTF-8

from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.models import Model
from keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, GlobalMaxPooling2D, Activation
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import layer_utils
from keras import backend as K
from Modules.Enums import FeatureExtractionModes as FE
import time

import cv2
import numpy as np


# CNN VGG 16 variables
_path_cnn_16_weights    = 'C:/VGG/rcmalli_vggface_tf_v2.h5'
_vgg16_model            = None

# specifies if actions and times should be logged or not
_is_logging             = False


def VGGArchitecture(include_top=True,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=2622
                    ):
    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    network = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(img_input)
    network = Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(network)
    network = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(network)

    # Block 2
    network = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(network)
    network = Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(network)
    network = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(network)

    # Block 3
    network = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(network)
    network = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(network)
    network = Convolution2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(network)
    network = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(network)

    # Block 4
    network = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(network)
    network = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(network)
    network = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(network)
    network = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(network)

    # Block 5
    network = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(network)
    network = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(network)
    network = Convolution2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(network)
    network = MaxPooling2D((2, 2), strides=(2, 2), name='pool5')(network)

    if include_top:
        # Classification block
        network = Flatten(name='flatten')(network)
        network = Dense(4096, activation='relu', name='fc6')(network)
        # network = Activation('relu', name='fc6/relu')(network)
        network = Dense(4096, activation='relu', name='fc7')(network)
        # network = Activation('relu', name='fc7/relu')(network)
        network = Dense(classes, activation='softmax', name='fc8')(network)
        # network = Activation('relu', name='fc8/softmax')(network)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(network)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(network)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, network, name='VGGFace')  # load weights
    return model


def _load_model(model, model_path, include_top=True):
    model.load_weights(model_path, by_name=True)
    if K.backend() == 'theano':
        layer_utils.convert_all_kernels_in_model(model)

    if K.image_data_format() == 'channels_first':
        if include_top:
            maxpool = model.get_layer(name='pool5')
            shape = maxpool.output_shape[1:]
            dense = model.get_layer(name='fc6')
            layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
    return model


def _extract_vgg16(img):
    global _vgg16_model, _path_cnn_16_weights

    if not _vgg16_model:
        loading_time = None
        if _is_logging:
            print('loading model')
            loading_time = time.time()

        _vgg16_model = VGGArchitecture()
        _vgg16_model = _load_model(_vgg16_model, _path_cnn_16_weights)

        if _is_logging:
            print('loading model:\t\t', time.time()-loading_time, 's')

    # calculates feature of input grayscale face image
    new_image = np.zeros((224, 224, 3), np.float32)

    new_image[:, :, 0] = img
    new_image[:, :, 1] = img
    new_image[:, :, 2] = img

    # unused variable
    # ret_val = False

    data = []
    new_image[:, :, 0] -= 93.5940
    new_image[:, :, 1] -= 104.7624
    new_image[:, :, 2] -= 129.1863

    new_image = np.transpose(new_image, (2, 0, 1))
    new_image = np.expand_dims(new_image, axis=0)
    new_image = new_image.flatten()

    data.append(new_image)
    data = np.array(data)

    if K.backend() == "theano":
        data = data.reshape(data.shape[0], 3, 224, 224)

    elif K.backend() == "tensorflow":
        data = data.reshape(data.shape[0], 224, 224, 3)

    data = data.astype('float32')
    data /= 255

    extract_feature = K.function([_vgg16_model.layers[0].input, K.learning_phase()],
                                 [_vgg16_model.layers[20].output])  # 20 is layer number

    return extract_feature([data])[0][0]


# ----------------------------------------------------------------------------------------------------------------------


def extract_features(img, mode=FE.CNN_VGG_16_PRE_TRAINED):
    """
    Takes given image and extract features with given method.

    For FeatureExtractionModes.CNN_VGG_16_PRE_TRAINED:
        image size has to be 224x224 pixel

    :param img:     grayscale image with correct size
    :param mode:    feature extraction mode of type Enums.FeatureExtractionModes
    :return:        multidimensional vector containing features of given image
    """
    start_time  = None
    features    = None
    if _is_logging:
        print('-----STARTING FEATURE EXTRACTION-----')
        print('backend:\t\t', K.backend())
        print('image shape:\t\t', img.shape)
        start_time = time.time()

    # extract features using VGG16 CNN with pre-trained weights
    if mode is FE.CNN_VGG_16_PRE_TRAINED:
        if _is_logging:
            print('Using pre-trained VGG16')

        # test if the image-shape is correct (224x224 needed)
        if img.shape[0] is not 224 or img.shape[1] is not 224:
            print('ERROR: FaceFeature.extract_features: Provided incorrect image shape!')
            print('given:\t', img.shape)
            print('needed:\t', '(224, 224')
            return

        # extract actual features
        features = _extract_vgg16(img)

    if _is_logging:
        print('total time:\t\t', time.time()-start_time, 's')

    return features


# SETTER
def set_logging(b):
    """
    True:   log actions and time
    False:  do not log anything

    :param b: boolean
    """
    global _is_logging
    if b:
        _is_logging = True
    if not b:
        _is_logging = False


# for testing purpose
if __name__ == '__main__':
    img_path = '../data/test/side/side1.jpg'
    img_ = cv2.imread(img_path, 0)
    img_ = cv2.resize(img_, (224, 224))
    extract_features(img_)


