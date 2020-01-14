from matplotlib import pylab as pylab
from skimage.io import imread

from skimage.color import rgb2gray, rgb2hsv, rgb2ycbcr, hsv2rgb, ycbcr2rgb
from skimage.feature import corner_harris, corner_subpix, corner_peaks, shape_index
from skimage.transform import warp, SimilarityTransform, AffineTransform, resize
import cv2
import numpy as np
from skimage import data
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import ransac

image = imread('imagens/imagem_8_seg.jpg')
image_gray = rgb2gray(image)

coordinates = corner_harris(image_gray, k=0.001)

#print(coordinates.max())
#print(0.001*coordinates.max())
#print(image_gray.shape)
linha, coluna = image_gray.shape
#for l in range(linha):
#    print(image_gray[l])

#image[coordinates > 0.01 * coordinates.max()] = [255, 0, 0, 255]


pylab.figure(figsize=(10, 5))
#pylab.axis('off')
pylab.imshow(image_gray, cmap='gray')
pylab.show()






print('FIM')