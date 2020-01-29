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

train_path_lagarta = 'imagens/treinamento/lagarta'
train_names_lagarta = os.listdir(train_path_lagarta)
feature_vector = []

for imagem in train_names_lagarta:
    for file in glob.glob(train_path_lagarta + '/' + imagem):
        image_rgb = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        fd, image_hog = hog(image_rgb, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        linha, coluna = image_hog.shape
        lista = []
        for l in range(linha):
            for c in range(coluna):
                if image_hog[l, c] > 0:
                    lista.append(image_hog[l, c])
                    #feature_vector.append(image_hog[l, c])
                    feature_vector.append(lista)


#hog_descriptor = cv2.HOGDescriptor()
#print(hog_descriptor)
#hog_people = cv2.HOGDescriptor_getDefaultPeopleDetector()
#print(hog_people)
print(feature_vector)



#print(vetor)
#pylab.plot(hog_descriptor)
#pylab.show()




print('FIM')