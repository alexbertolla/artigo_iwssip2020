import numpy as np
import os
from sklearn.svm import LinearSVC


def treinamento(trainning_features_list, trainning_class_list):
    print('Training features: {}'.format(np.array(trainning_features_list).shape))
    print('Training labels: {}'.format(np.array(trainning_class_list).shape))

    print('[STATUS] Creating the classifier..')
    clf_svm = LinearSVC(random_state=9)

    print('[STATUS] Fitting data/label to model..')
    clf_svm.fit(trainning_features_list, trainning_class_list)
    return clf_svm
