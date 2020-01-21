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


def ler_imagens_treinamento(diretorio, lista_imagens, classe):
    array_hog = []
    array_classe = []
    for imagem in lista_imagens:
        for file in glob.glob(diretorio + '/'+imagem):
#            print('Processing Image {}'. format(imagem))
            image_rgb = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
#            gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
            fd, image_hog = hog(image_rgb, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
            linha, coluna = image_hog.shape
            for l in range(linha):
                for c in range(coluna):
                    if image_hog[l, c] > 0:
                        train_features.append(image_hog[l, c])
                        train_labels.append(classe)


classe_spodoptera = 'Spodoptera'
classe_outra_lagarta = 'Outra lagarta'

train_path_lagarta = 'imagens/treinamento/lagarta'
train_names_lagarta = os.listdir(train_path_lagarta)

train_path_outra_lagarta = 'imagens/treinamento/outra_lagarta'
train_names_outra_lagarta = os.listdir(train_path_outra_lagarta)

train_features = []
train_labels = []

array_caracteristicas = []

print('[STATUS] Started extracting haralick textures from Spodoptera images..')

#fig, axes = pylab.subplots(ncols=3, nrows=4)
x = 0
y = 0
#LER IMAGENS TREINAMENTO SPODOPTERA FUNGIPERD

print(train_features)
print(train_labels)


ler_imagens_treinamento(train_path_outra_lagarta, train_names_outra_lagarta, classe_outra_lagarta)
#ler_imagens_treinamento(train_path_lagarta, train_names_lagarta, classe_spodoptera)


print(train_labels)
print(train_features)


#array_caracteristicas_outras_lagartas = ler_imagens_treinamento(train_path_outra_lagarta, train_names_outra_lagarta, classe_outra_lagarta)
#for train_name_lagarta in train_names_lagarta:
#    cur_path = train_path_lagarta + '/' + train_name_lagarta
#    cur_label = train_name_lagarta

#    for file in glob.glob(cur_path):
#        print('Processing Image - {} in {}'. format(i, cur_label))
#        image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
#        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#        fd, image_hog = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
#        array_hog = []
#        linha, coluna = image_hog.shape
#        for l in range(linha):
#            for c in range(coluna):
#                if image_hog[l, c] > 0:
#                    array_hog.append([cur_label, l, c, image_hog[l, c]])
#       array_imagem_hog.append(array_hog)

#print(array_imagem_hog)
#        pylab.figure(figsize=(4, 4))
#        pylab.imshow(image_hog)
        #train_features.append(features)
        #train_labels.append(classe_spodoptera)

#pylab.show()

#fd, image_8_hog = hog(image_8, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
#fd, image_8_original_hog = hog(image_8_original, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

#fd, image_67_hog = hog(image_67, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
#fd, image_77_hog = hog(image_77, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
#fd, image_88_hog = hog(image_88, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)


#fig, (axes11, axes12, axes13, axes14) = pylab.subplots(1, 4, figsize=(15, 10), sharex=True, sharey=True)
#fig, (axes21, axes22, axes23, axes24) = pylab.subplots(1, 4, figsize=(15, 10), sharex=True, sharey=True)

#fig, plot = pylab.subplots(3, 6)

#image_8_hog_rescaled = exposure.rescale_intensity(image_8_hog, in_range=(0, 10))
#image_8_original_hog_rescaled = exposure.rescale_intensity(image_8_original_hog, in_range=(0, 10))

#image_67_hog_rescaled = exposure.rescale_intensity(image_67_hog, in_range=(0, 10))
#image_77_hog_rescaled = exposure.rescale_intensity(image_77_hog, in_range=(0, 10))
#image_88_hog_rescaled = exposure.rescale_intensity(image_88_hog, in_range=(0, 10))




print('FIM')