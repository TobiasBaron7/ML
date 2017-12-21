###FACE DETECTION - FACE RECOGNITION###
import cv2
from Modules import FaceExtractor as FE
from Modules import FaceFeature as FF
from Modules import FaceIdentificator as FI
from Modules.Enums import ImageProcessingModes as Ip
from Modules.Enums import FaceLocalisationModes as Fl
from Modules.Enums import FeatureExtractionModes as Fe
import pickle
from PIL import ImageGrab
import os
import time
from tqdm import tqdm
# import random


test_data       = 'test_data.txt'
data            = None
data_size       = 0


def finish(name='screenshot'):
    """
    Takes screenshot as .jpeg and shuts down the pc.

    :param name: name of the screenhot
    """
    try:
        name = str(name)
    except ValueError or TypeError:
        name = 'screenshot'
    ImageGrab.grab().save(name + '.jpeg', format='JPEG', subsampling=0, quality=100)
    os.system('shutdown -s')


def init(test_data_path=None):
    """
    Loads .txt with paths to test-images and prepares database for FaceIdentification

    :param test_data_path: path to .txt with test-images, default 'test_data.txt'
    """
    global data, test_data, data_size

    FI.get_ready()

    if test_data_path:
        test_data = test_data_path
    print('Loading file:\t\t', test_data)
    with open(test_data, 'rb') as fp:
        data = pickle.load(fp)
    data_size = len(data)
    print('Loaded test-images:\t', data_size)


def test(pre_cliplimit=5, pre_tile_grid_size=(8, 8), scale_factor=1.3, min_neighbors=5,
         max_faces=1, face_out_size=(224, 224)):
    global data, data_size

    # Meta data
    start_time          = time.time()
    true_positive       = 0
    false_positive      = 0
    avg_dist_true_pos   = 0
    avg_dist_false_pos  = 0
    correct_data        = {}    # dict with image_info as key and 2D-array as value
    incorrect_data      = {}    # with two numbers: [cosine-distance, number of comparisons]

    # only for testing: replace data with random_sample in for-loop
    # sample_size = 15
    # random_sample = random.sample(data, sample_size)
    counter = 0
    for image_path in tqdm(data):
        counter += 1
        # image_path is like 'C:/toolkits/databases/ExtendedYaleB/yaleB39/yaleB39_P08A-035E+15.pgm'
        image_info = image_path.split('/')[-1:][0][:-4]
        person_id = int(image_info[5:7])
        # it's starting at 11 and number 14 is missing
        person_id = person_id - 10 if person_id < 15 else person_id - 11
        img = cv2.imread(image_path, 0)
        face_list = FE.extract_faces(img,
                                     [{Ip.CLAHE:
                                           {'cliplimit': pre_cliplimit,
                                            'tile_grid_size': pre_tile_grid_size}}],
                                     [{Fl.HAARCASCADES_FACE_PRE_TRAINED:
                                           {'scale_factor': scale_factor,
                                            'min_neighbors': min_neighbors,
                                            'max_faces': max_faces}}],
                                     face_out_size=face_out_size)
        # extract features of the face
        if face_list and len(face_list) > 0:
            face_feature = FF.extract_features(face_list[0], mode=Fe.CNN_VGG_16_PRE_TRAINED)
            try:
                match_row, min_d, num_comp = FI.identify(face_feature)
                match_id = match_row[1]

                if match_id is person_id:
                    correct_data[image_info] = [min_d, num_comp]
                    true_positive += 1
                    avg_dist_true_pos += min_d
                else:
                    incorrect_data[image_info] = [min_d, num_comp]
                    false_positive += 1
                    avg_dist_false_pos += min_d
            except:
                print('ERROR: No FaceFeature for', image_path)
                pass

    total_time = time.time() - start_time

    avg_dist_true_pos = avg_dist_true_pos/true_positive if true_positive > 0 else 0
    avg_dist_false_pos = avg_dist_false_pos/false_positive if false_positive > 0 else 0

    print('Elapsed time:', total_time, 's')
    print('Time per image:', total_time/data_size, 's')
    print('\ntrue-positive:', true_positive)
    print('false-positive:', false_positive)
    print('\navg_dist_true:', avg_dist_true_pos)
    print('avg_dist_false:', avg_dist_false_pos)

    with open('database_test_results.txt', 'w') as f:
        f.write('Datasize total: ' + str(data_size))
        f.write('\nDatasize tested: ' + str(true_positive + false_positive))
        f.write('\nElapsed time: ' + str(total_time) + 's')
        f.write('\nTime per image: ' + str(total_time/data_size) + 's')
        f.write('\n\ntrue-positive: ' + str(true_positive))
        f.write('\nfalse-positive: ' + str(false_positive))
        f.write('\n\navg_dist_true: ' + str(avg_dist_true_pos))
        f.write('\navg_dist_false: ' + str(avg_dist_false_pos))

    with open('correct_classification_stats.pkl', 'wb') as co:
        pickle.dump(correct_data, co)
    with open('incorrect_classification_stats.pkl', 'wb') as ico:
        pickle.dump(incorrect_data, ico)


if __name__ == '__main__':
    # if True: take screenshot and shutdown when error occurs
    error_shutdown = False
    # if True: shutdown when test is done
    finish_shutdown = False

    try:
        init()
        # test automatically saves results to disk after done
        test()
    except:
        if error_shutdown:
            finish()
        else:
            raise
    if finish_shutdown:
        finish()

