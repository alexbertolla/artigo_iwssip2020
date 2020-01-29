from matplotlib import pylab as pylab
from skimage.io import imread, imsave, imshow
from skimage import exposure
from skimage.color import rgb2gray, rgb2hsv, rgb2ycbcr, hsv2rgb, ycbcr2rgb, gray2rgb, rgb2xyz
from skimage.feature import corner_harris, corner_subpix, corner_peaks, shape_index, hog
from skimage.transform import warp, SimilarityTransform, AffineTransform, resize
import cv2
import numpy as np
from numpy import array
from skimage import data
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import ransac
from PIL import Image
import os
import glob
from sklearn.svm import LinearSVC
import mahotas as mt


def extract_features(image, image_label):
    textures = mt.features.haralick(image)
    #read_textures(textures, image_label)
    ht_mean = textures.mean(axis=0)
    return ht_mean

def ler_imagens_treinamento(diretorio, lista_imagens, classe):
    array_hog = []
    array_classe = []
    for imagem in lista_imagens:
        for file in glob.glob(diretorio + '/'+imagem):
#            print('Processing Image {}'. format(imagem))
            image_rgb = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            features = extract_features(gray, imagem)
            feature_vector_haralick.append(features)
            label_haralick.append(classe)

            fd, image_hog = hog(image_rgb, orientations=3, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            cv2.imshow(imagem, image_hog)
            cv2.waitKey(0)

            linha, coluna = image_hog.shape

            for l in range(linha):
                for c in range(coluna):
                    if image_hog[l, c] > 0:
                        train_features.append(image_hog[l, c])
                        train_labels.append(classe)

def resultado_svm(svm_class):
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
            cv2.imshow(cur_label, image)

            # convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # extract haralick texture from the image
            features = extract_features(gray, cur_label)

            # evaluate the model and predict label
            prediction = svm_class.predict(features.reshape(1, -1))[0]

            # show the label
            cv2.putText(image, prediction, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            print('Prediction - {}'.format(prediction))

            # display the output image
            cv2.imshow(cur_label, image)
            cv2.waitKey(0)

classe_spodoptera = 'Spodoptera'
classe_outra_lagarta = 'Outra lagarta'
classe_nao_lagarta = 'Nao lagarta'

train_path_lagarta = 'imagens/treinamento/lagarta'
train_names_lagarta = os.listdir(train_path_lagarta)

train_path_outra_lagarta = 'imagens/treinamento/outra_lagarta'
train_names_outra_lagarta = os.listdir(train_path_outra_lagarta)

train_path_nao_lagarta = 'imagens/treinamento/nao_lagarta'
train_names_nao_lagarta = os.listdir(train_path_nao_lagarta)

train_features = []
train_labels = []

feature_vector_haralick = []
label_haralick = []

array_caracteristicas = []

print('[STATUS] Started extracting haralick textures from Spodoptera images..')

#fig, axes = pylab.subplots(ncols=3, nrows=4)
x = 0
y = 0
#LER IMAGENS TREINAMENTO SPODOPTERA FUNGIPERD


ler_imagens_treinamento(train_path_lagarta, train_names_lagarta, classe_spodoptera)
#ler_imagens_treinamento(train_path_nao_lagarta, train_names_nao_lagarta, classe_nao_lagarta)


print('[STATUS] Creating the classifier..')
clf_svm = LinearSVC(random_state=0 )

print('[STATUS] Fitting data/label to model..')
clf_svm.fit(feature_vector_haralick, label_haralick)

#resultado_svm(clf_svm)

print('FIM')