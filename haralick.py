import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
from svm import treinamento

def read_textures(textures, image_label):
    linha, coluna = textures.shape
    cv2.imshow(image_label, textures)
    for l in range(linha):
        for c in range(coluna):
            print(textures[l, c])


def extract_features(image, image_label):
    textures = mt.features.haralick(image)
    #read_textures(textures, image_label)
    ht_mean = textures.mean(axis=0)
    return ht_mean


train_path_lagarta = 'imagens/treinamento/lagarta'
train_names_lagarta = os.listdir(train_path_lagarta)

train_path_outra_lagarta = 'imagens/treinamento/outra_lagarta'
train_names_outra_lagarta = os.listdir(train_path_outra_lagarta)

classe_spodoptera = 'Spodoptera'
classe_outra_lagarta = 'Outra lagarta'

train_features = []
train_labels = []

print('[STATUS] Started extracting haralick textures from Spodoptera images..')
for train_name_lagarta in train_names_lagarta:
    cur_path = train_path_lagarta + '/' + train_name_lagarta
    cur_label = train_name_lagarta
    i = 1
    for file in glob.glob(cur_path):
        print('Processing Image - {} in {}'. format(i, cur_label))
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = extract_features(gray, cur_label)
        train_features.append(features)
        train_labels.append(classe_spodoptera)
        i += 1

print('[STATUS] Started extracting haralick textures from others images..')
for train_name_outra_lagartalagarta in train_names_outra_lagarta:
    cur_path = train_path_outra_lagarta + '/' + train_name_outra_lagartalagarta
    cur_label = train_name_outra_lagartalagarta
    i = 1
    for file in glob.glob(cur_path):
        print('Processing Image - {} in {}'. format(i, cur_label))
        image = cv2.imread(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = extract_features(gray, cur_label)
        train_features.append(features)
        train_labels.append(classe_outra_lagarta)
        i += 1


print('Training features: {}'.format(np.array(train_features).shape))
print('Training labels: {}'.format(np.array(train_labels).shape))

print('[STATUS] Creating the classifier..')
clf_svm = LinearSVC(random_state=9)

print('[STATUS] Fitting data/label to model..')
clf_svm.fit(train_features, train_labels)


print('[STATUS] Starting tests..')
test_path = "imagens/teste"
test_names = os.listdir(test_path)
for test_name in test_names:
    cur_path = test_path + '/' + test_name
    cur_label = test_name
    i = 1
    for file in glob.glob(cur_path):

        # read the input image
        image = cv2.imread(file)

        #cv2.imshow(cur_label, image)

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(gray, cur_label)

        # evaluate the model and predict label
        prediction = clf_svm.predict(features.reshape(1, -1))[0]

        # show the label
        cv2.putText(image, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        print('Prediction - {}'.format(prediction))

        # display the output image
        cv2.imshow(cur_label, image)
        cv2.waitKey(0)


print('FIM')