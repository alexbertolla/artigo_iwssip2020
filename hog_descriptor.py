from matplotlib import pylab as pylab
from skimage.io import imread
from skimage import exposure
from skimage.color import rgb2gray, rgb2hsv, rgb2ycbcr, hsv2rgb, ycbcr2rgb
from skimage.feature import corner_harris, corner_subpix, corner_peaks, shape_index, hog
from skimage.transform import warp, SimilarityTransform, AffineTransform, resize
import cv2
import numpy as np
from skimage import data
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import ransac
from PIL import Image

image = imread('imagens/imagem_77_seg.jpg')
image2 = imread('imagens/imagem_77.jpg')

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
#fd2, hog_image2 = hog(image2, orientations=20, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

linha, coluna, dimensao = image.shape
pixel_max = 0
pixel_min = 0
nova_imagem = np.zeros([256, 256])

for l in range(linha):
    for c in range(coluna):
#        print(image[l, c, 0])
#        nova_imagem[l, c] = 1.0
#        print(hog_image[l, c])
        if image[l, c, 0] < 255:
            image2[l, c] = 0
        else:
            image2[l, c] = 128

fig, (axes1, axes2, axes3) = pylab.subplots(1, 3, figsize=(15, 10), sharex=True, sharey=True)
axes1.axis('off'), axes1.imshow(image), axes1.set_title('Imput Image')
#hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
#axes2.axis('off'), axes2.imshow(hog_image_rescaled, cmap=pylab.cm.gray), axes2.set_title('Histogram of Oriented Gradients')
axes3.axis('off'), axes3.imshow(image2), axes3.set_title('TESTE')


#pylab.figure(figsize=(10, 5))
#pylab.axis('off')
#pylab.imshow(nova_imagem, cmap='gray')

pylab.show()


print('FIM')