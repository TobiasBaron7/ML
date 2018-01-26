import os
import numpy as np
import scipy
import pickle
import ctypes
from tqdm import trange
import cv2
from Modules import FaceExtractor as FE
from Modules.Enums import ImageProcessingModes as Ip
from Modules.Enums import FaceLocalisationModes as Fl

_PATH_ORIGIN        = 'C:/toolkits/databases/ExtendedYaleB_jpg'
_OUTPUT_DIR         = 'Original_Yale_Cropped'

_NUM_YALE_ID        = 28

# ImageProcessor SETTINGS
pre_cliplimit       = 5
pre_tile_grid_size  = (8, 8)
face_out_size       = (224, 224)

# haarcascades SETTINGS
scale_factor        = 1.3
min_neighbors       = 5
max_faces           = 1

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

    id_counter = 0
    for sub_folder in os.listdir(_PATH_ORIGIN):
        create_dir_if_not_exist(_OUTPUT_DIR + '/' + sub_folder)
        for file in os.listdir(_PATH_ORIGIN + '/' + sub_folder):
            file_name = file[:-4]
            images_counter += 1
            image = cv2.imread(_PATH_ORIGIN + '/' + sub_folder + '/' + file, 0)
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
                scipy.misc.imsave(_OUTPUT_DIR + '/' + sub_folder + '/' + file_name + '.png', face[0])
                success_counter += 1
                if id_counter in images_success:
                    images_success[id_counter].append(file_name)
                else:
                    images_success[id_counter] = [file_name]

            else:
                failed_counter += 1
                if id_counter in images_failed:
                    images_failed[id_counter].append(file_name)
                else:
                    images_failed[id_counter] = [file_name]

        id_counter += 1



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
        #system_shutdown()
    except Exception as e:
        save_data()
        print(e)
        with open('ERROR_LOG.txt', 'w') as error_log:
            error_log.write(str(e))
        #system_shutdown()

