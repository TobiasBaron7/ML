import os
from shutil import copyfile
import pickle


_PATH_ORIGIN    = 'Original_Yale_Cropped_Pose_0-5'

images_failed = {}
images_success = {}


id_counter = 0
for sub_folder in os.listdir(_PATH_ORIGIN):
    for file in os.listdir(_PATH_ORIGIN + '/' + sub_folder):
        if id_counter in images_success:
            images_success[id_counter].append(file)
        else:
            images_success[id_counter] = [file]
    id_counter += 1

with open('images_success', 'wb') as _:
    pickle.dump(images_success, _)



