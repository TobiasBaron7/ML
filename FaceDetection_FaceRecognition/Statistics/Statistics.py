import pickle
import numpy as np
import Plotter as plotter
# Plotly is not used anymore, only for html-heatmap
# import Plotly


_PATH_TEST_DATA         = 'Data/test_data.txt'
_PATH_TRAIN_DATA        = 'Data/training_data.txt'
_PATH_VALID_DATA        = 'Data/valid_data.txt'
_PATH_CORRECT_TEST      = 'Data/correct_classification_stats.pkl'
_PATH_INCORRECT_TEST    = 'Data/incorrect_classification_stats.pkl'
_PATH_CONFUSION_MATRIX  = 'Data/29/confusion_matrix.npy'
_PATH_INPUT_TO_OUTPUT   = 'Data/input_info_to_output_feature_identity.pkl'


class Statistic:
    def __init__(self):
        """
        with open(_PATH_TEST_DATA, 'rb') as _:
            self.DATA_TEST              = pickle.load(_)
        with open(_PATH_TRAIN_DATA, 'rb') as _:
            self.DATA_TRAIN             = pickle.load(_)
        with open(_PATH_VALID_DATA, 'rb') as _:
            self.VALID_DATA             = pickle.load(_)
        with open(_PATH_CORRECT_TEST, 'rb') as _:
            self.DATA_CORRECT           = pickle.load(_)
        with open(_PATH_INCORRECT_TEST, 'rb') as _:
            self.DATA_INCORRECT         = pickle.load(_)
        with open(_PATH_INPUT_TO_OUTPUT, 'rb') as _:
            self.DATA_INPUT_TO_OUTPUT   = pickle.load(_)
        """

        self.confusion_matrix       = np.load(_PATH_CONFUSION_MATRIX)
        self.confusion_classes      = []
        self.confusion_data         = np.empty(shape=(28, 28))
        self.confusion_data_normed  = np.empty(shape=(28, 28))
        for i in range(len(self.confusion_matrix)-1):
            self.confusion_classes.append(str(i + 1))
        for i in range(len(self.confusion_matrix)):
            if i == 0:
                continue
            self.confusion_data[i - 1] = self.confusion_matrix[i][1:]
            #self.confusion_data_normed[i - 1] = np.around((self.confusion_matrix[i][1:] / np.linalg.norm(self.confusion_matrix[i][1:]) * 100), decimals=4)
            self.confusion_data_normed[i - 1] = self.confusion_matrix[i][1:]

        self.true_positive      = 3072#len(self.DATA_CORRECT)
        self.false_positive     = 2019#len(self.DATA_INCORRECT)

        self.identities         = {}

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


if __name__ == '__main__':
    statistic = Statistic()

    # deprecated use of plotly heatmap
    # Plotly.plot_heatmap(statistic.confusion_x, statistic.confusion_x, statistic.confusion_data_normed)

    plotter.plot_confusion_matrix(statistic.confusion_data, classes=statistic.confusion_classes, normalize=False, title='29')

