import numpy as np
import pickle
import matplotlib.pyplot as plt


_PATH_SUCCESS   = 'Logs/Artificial_Yale/images_success'
_PATH_FAILED    = 'Logs/Artificial_Yale/images_failed'


with open(_PATH_SUCCESS, 'rb') as _:
    images_success = pickle.load(_)

with open(_PATH_FAILED, 'rb') as _:
    images_failed = pickle.load(_)


def plot_histogram(data, normed, label):
    plt.hist(data, normed=normed, bins=len(data))
    plt.ylabel(label)
    plt.show()


def plot_bar_chart(data, xlabel, ylabel, info=None, label=None):
    if info:
        print('Plotting:', info)
    plt.bar(np.arange(len(data)), height=data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label and len(label) is len(data):
        plt.xticks(np.arange(len(data)), label)
    plt.show()


data_failed_per_identity    = []
data_success_per_identity   = []

data_failed_poses           = [0] * 6
data_success_poses          = [0] * 6

data_failed_azimuth_d       = {}
data_success_azimuth_d      = {}
data_failed_azimuth         = []
data_success_azimuth        = []
label_azimuth               = []
label_azimuth_s             = []

data_failed_elevation_d     = {}
data_success_elevation_d    = {}
data_failed_elevation       = []
data_success_elevation      = []
label_elevation             = []
label_elevation_s           = []

for id in images_failed:
    id_img_list = images_failed[id]

    for image in id_img_list:
        pose = int(image[1:3])
        azimuth = int(image[image.index('A') + 1:image.index('E')])
        elevation = int(image[image.index('E') + 1:])
        if azimuth in data_failed_azimuth_d.keys():
            data_failed_azimuth_d[azimuth] += 1
        else:
            data_failed_azimuth_d[azimuth] = 1

        if elevation in data_failed_elevation_d.keys():
            data_failed_elevation_d[elevation] += 1
        else:
            data_failed_elevation_d[elevation] = 1

        data_failed_poses[pose] += 1

    data_failed_per_identity.append(len(images_failed[id]))

for id in images_success:
    id_img_list = images_success[id]

    for image in id_img_list:
        pose = int(image[1:3])
        azimuth = int(image[image.index('A') + 1:image.index('E')])
        elevation = int(image[image.index('E') + 1:])
        if azimuth in data_success_azimuth_d.keys():
            data_success_azimuth_d[azimuth] += 1
        else:
            data_success_azimuth_d[azimuth] = 1

        if elevation in data_success_elevation_d.keys():
            data_success_elevation_d[elevation] += 1
        else:
            data_success_elevation_d[elevation] = 1

        data_success_poses[pose] += 1

    data_success_per_identity.append(len(images_success[id]))


for k in data_failed_azimuth_d.keys():
    data_failed_azimuth.append(k)
for k in data_success_azimuth_d.keys():
    data_success_azimuth.append(k)
label_azimuth = sorted(data_failed_azimuth)
label_azimuth_s = sorted(data_success_azimuth)
for k in data_failed_elevation_d.keys():
    data_failed_elevation.append(k)
for k in data_success_elevation_d.keys():
    data_success_elevation.append(k)
label_elevation = sorted(data_failed_elevation)
label_elevation_s = sorted(data_success_elevation)

for i in range(len(label_azimuth)):
    data_failed_azimuth[i] = data_failed_azimuth_d[label_azimuth[i]]
for i in range(len(label_azimuth_s)):
    data_success_azimuth[i] = data_success_azimuth_d[label_azimuth_s[i]]
for i in range(len(label_elevation)):
    data_failed_elevation[i] = data_failed_elevation_d[label_elevation[i]]
for i in range(len(label_elevation_s)):
    data_success_elevation[i] = data_success_elevation_d[label_elevation_s[i]]

# PLOTTING
print('PATH:\t', _PATH_FAILED, '\n\t', _PATH_SUCCESS, '\n')

plot_bar_chart(data_failed_azimuth, xlabel='Azimuth', ylabel='Number of bad images', info='data failed azimuth', label=label_azimuth)
plot_bar_chart(data_success_azimuth, xlabel='Azimuth', ylabel='Number of good images', info='data success azimuth', label=label_azimuth_s)
plot_bar_chart(data_failed_elevation, xlabel='Elevation', ylabel='Number of bad images', info='data failed elevation', label=label_elevation)
plot_bar_chart(data_success_elevation, xlabel='Elevation', ylabel='Number of good images', info='data success elevation', label=label_elevation_s)

#plot_bar_chart(data_failed_per_identity, xlabel='Identity', ylabel='Number of images', info='data failed per identity')
#plot_bar_chart(data_success_per_identity, xlabel='Identity', ylabel='Number of images', info='data success per identity')

#plot_bar_chart(data_failed_poses, xlabel='Pose', ylabel='Number of images', info='artificial failed poses')
#plot_bar_chart(data_success_poses, xlabel='Pose', ylabel='Number of images', info='artificial success poses')

