import os
import random
from shutil import copyfile


_PATH_ORIGIN    = 'Artificial_Yale_Extension'
_PATH_DEST      = 'Random_Sample'
_NUM_SAMPLES    = 585

_SEED           = 0


random.seed(_SEED)
all_files = []

for sub_folder in os.listdir(_PATH_ORIGIN):
    for file in os.listdir(_PATH_ORIGIN + '/' + sub_folder):
        all_files.append(_PATH_ORIGIN + '/' + sub_folder + '/' + file)


random_sample = [all_files[i] for i in random.sample(range(len(all_files)), _NUM_SAMPLES)]

if not os.path.exists(_PATH_DEST):
    os.makedirs(_PATH_DEST)

for file in random_sample:
    file_name = str(file).split('/')[2]
    copyfile(file, _PATH_DEST + '/' + file_name)

