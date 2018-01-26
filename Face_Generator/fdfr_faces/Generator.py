import os
from keras.models import load_model
import numpy as np
import scipy
import pickle
import ctypes
from tqdm import trange
from Modules import FaceExtractor as FE
from Modules.Enums import ImageProcessingModes as Ip
from Modules.Enums import FaceLocalisationModes as Fl

_MODEL_PATH         = 'C:/Users/FinalFred/Documents/SourceTree-Projects/ML/Face_Generator/deconvfaces/output/' \
                      '100_epochs_tensorflow_FaceGen.YaleFaces.model.d5.adam.h5'
_OUTPUT_DIR         = 'Artificial_Yale_Extension'

_NUM_YALE_ID        = 28
_NUM_YALE_POSES     = 10
_NUM_YALE_LIGHT     = 4

# GENERATOR SETTINGS
_ARTIFICIAL_POSES   = [0, 1, 2, 3, 4, 5]
# key = azimuth, value = elevation
# same as in original yale b extended
_ARTIFICIAL_LIGHT   = {-130: [20],
                       -120: [0],
                       -110: [15, 40, 65, -20],
                       -95: [0],
                       -85: [20, -20],
                       -70: [0, 45, -35],
                       -60: [20, -20],
                       -50: [0, -40],
                       -35: [15, 40, 65, -20],
                       -25: [0],
                       -20: [10, -10, -40],
                       -15: [20],
                       -10: [0, -20],
                       -5: [10, -10],
                       0: [0, 20, 45, 90, -20, -35],
                       5: [10, -10],
                       10: [0, -20],
                       15: [20],
                       20: [10, -10, -40],
                       25: [0],
                       35: [15, 40, 65, -20],
                       50: [0, -40],
                       60: [20, -20],
                       70: [0, 45, -35],
                       85: [20, -20],
                       95: [0],
                       110: [15, 40, 65, -20],
                       120: [0],
                       130: [20]
                       }

# ImageProcessor SETTINGS
pre_cliplimit       = 5
pre_tile_grid_size  = (8, 8)
face_out_size       = (224, 224)

# haarcascades SETTINGS
scale_factor        = 1.3
min_neighbors       = 5
max_faces           = 1

model               = load_model(_MODEL_PATH)

id_vector           = [0.] * _NUM_YALE_ID
light_vector        = [0.] * _NUM_YALE_LIGHT
pose_vector         = [0]  * _NUM_YALE_POSES

# meta data for statistics
images_counter = 0
failed_counter = 0
success_counter = 0
images_failed = {}
images_success = {}


def create_dir_if_not_exist(path):
    """
    Checks if given directory already exists.
    If not, creates this directory.

    :param path:  Directory to check
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_id_vector_from_id(start_id):
    """
    Takes an ID and return a mixed id-vector,
    consisting of 100% start_id and 50% start_id + 1.
    When last possible id is given, its mixed with first one. (27=100%, 0=50%)

    :param start_id: id of first person
    :return: id vector
    """
    global id_vector
    if start_id >= _NUM_YALE_ID:
        print('ERROR in Generator.get_id_vector_from_id: Given id is out of bounds!')
        return
    for i in range(len(id_vector)):
        id_vector[i] = 0
    v = id_vector
    v_ = []
    v[start_id] = 1
    if start_id < _NUM_YALE_ID - 1:
        v[start_id + 1] = 0.5
    else:
        v[0] = 0.5
    v_.append(v)
    return np.array(v_)


def get_pose_vector_from_pose(p):
    global pose_vector
    for i in range(len(pose_vector)):
        pose_vector[i] = 0
    v = pose_vector
    v_ = []
    v[p] = 1
    v_.append(v)
    return np.array(v_)


def get_lightning_vector_from_lightning(azimuth, elevation):
    """
    Takes given arguments and transforms them into a lightning-vetor,
    as expected from the deconvfaces-model.

    :param azimuth:     in range [-90, 90]
    :param elevation:   in range [-90, 90]
    :return:            vector (length=4) with lightning representation
    """
    azrad = np.deg2rad(azimuth)
    elrad = np.deg2rad(elevation)
    v = [np.sin(azrad), np.cos(azrad), np.sin(elrad), np.cos(elrad)]
    v_ = []
    v_.append(v)
    return np.array(v_)


def save_data():
    global images_counter, failed_counter, success_counter, images_failed, images_success
    with open('images_success', 'wb') as t:
        pickle.dump(images_success, t)
    with open('images_failed', 'wb') as t:
        pickle.dump(images_failed, t)

    print('END OF GENERATIONS')
    print('Total images:', images_counter)
    print('Successful:', success_counter)
    print('Failed:', failed_counter)


def main():
    global images_counter, failed_counter, success_counter, images_failed, images_success
    create_dir_if_not_exist(_OUTPUT_DIR)

    key_list_lightning = list(_ARTIFICIAL_LIGHT.keys())

    for i in trange(_NUM_YALE_ID, desc='Identities', position=0):
        folder_path = _OUTPUT_DIR + '/yaleB' + str(i + 40)
        create_dir_if_not_exist(folder_path)

        id_vec = get_id_vector_from_id(i)
        for p in trange(len(_ARTIFICIAL_POSES), desc='Poses', position=1):
            pose = _ARTIFICIAL_POSES[p]
            pose_vec = get_pose_vector_from_pose(pose)
            for a in trange(len(_ARTIFICIAL_LIGHT), desc='Azimuth', position=2):
                azimuth = key_list_lightning[a]
                elevation_list = _ARTIFICIAL_LIGHT[azimuth]
                for e in trange(len(elevation_list), desc='Elevation', position=3):
                    elevation = elevation_list[e]
                    light_vec = get_lightning_vector_from_lightning(azimuth, elevation)
                    image_meta = 'P{:02}A{:03}E{:02}'.format(pose, azimuth, elevation)

                    batch = {
                        'identity': id_vec,
                        'pose': pose_vec,
                        'lighting': light_vec,
                    }

                    generated_face = model.predict_on_batch(batch)

                    for j in range(0, generated_face.shape[0]):
                        images_counter += 1
                        image = generated_face[j, :, :, 0]
                        image = np.array(255 * np.clip(image, 0, 1), dtype=np.uint8)
                        file_path = folder_path + '/' + 'yaleB' + str(i + 40) + '_{}.{}'.format(image_meta, 'png')

                        face = FE.extract_faces(image,
                                             [{Ip.CLAHE:
                                                   {'cliplimit': pre_cliplimit,
                                                    'tile_grid_size': pre_tile_grid_size}}],
                                             [{Fl.HAARCASCADES_FACE_PRE_TRAINED:
                                                   {'scale_factor': scale_factor,
                                                    'min_neighbors': min_neighbors,
                                                    'max_faces': max_faces}}],
                                             face_out_size=face_out_size)

                        if face and len(face) == 1:
                            scipy.misc.imsave(file_path, face[0])
                            success_counter += 1
                            if i + 40 in images_success:
                                images_success[i + 40].append(image_meta)
                            else:
                                images_success[i + 40] = [image_meta]

                        else:
                            failed_counter += 1
                            if i + 40 in images_failed:
                                images_failed[i + 40].append(image_meta)
                            else:
                                images_failed[i + 40] = [image_meta]


def system_shutdown():
    from PIL import ImageGrab
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    name = 'screenshot'
    ImageGrab.grab().save(name + '.jpeg', format='JPEG', subsampling=0, quality=100)
    os.system('shutdown -s')


if __name__ == '__main__':
    'https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx'
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    try:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        main()
        save_data()
        system_shutdown()
    except Exception as e:
        save_data()
        print(e)
        with open('ERROR_LOG.txt', 'w') as error_log:
            error_log.write(e)
        system_shutdown()


