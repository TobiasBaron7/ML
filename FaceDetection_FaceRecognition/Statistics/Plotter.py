import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


cdict = {'red': ((0.0, 1.0, 1.0),
                           (0.001, 0.5, 0.5),
                           (0.05, 1.0, 1.0),
                           (0.11, 0.0, 0.0),
                           (1.0, 0.0, 0.0)),

             'green': ((0.0, 1.0, 1.0),
                       (0.001, 0.0, 0.0),
                       (0.5, 0.5, 0.5),
                       (1.0, 1.0, 1.0)),

             'blue': ((0.0, 1.0, 1.0),
                      (0.001, 0.0, 0.0),
                      (1.0, 0.0, 0.0))}

colormap_confusion = LinearSegmentedColormap('Confusion_Color', cdict)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=colormap_confusion,
                          colorbar=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    This is a modified version of an example from www.scikit-learn.org.
    Orignal: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.figure()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if colorbar:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.show()
