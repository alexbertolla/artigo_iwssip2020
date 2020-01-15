from matplotlib import pylab as pylab
from skimage.io import imread, imsave, imshow
from skimage import exposure
from skimage.color import rgb2gray, rgb2hsv, rgb2ycbcr, hsv2rgb, ycbcr2rgb, gray2rgb, rgb2xyz
from skimage.feature import corner_harris, corner_subpix, corner_peaks, shape_index, hog
from skimage.transform import warp, SimilarityTransform, AffineTransform, resize
import cv2
import numpy as np
from skimage import data
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from skimage.measure import ransac
from PIL import Image

from escala_cinza import rgbTogray

image_8_original = imread('imagens/imagem_51.jpg')
image_8 = imread('imagens/imagem_8_seg.jpg')
image_67 = imread('imagens/imagem_67_seg.jpg')
image_77 = imread('imagens/imagem_77_seg.jpg')
image_88 = imread('imagens/imagem_88_seg.jpg')

image_77_gray = rgb2gray(image_77)


fd, image_8_hog = hog(image_8, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
fd, image_8_original_hog = hog(image_8_original, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

fd, image_67_hog = hog(image_67, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
fd, image_77_hog = hog(image_77, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
fd, image_88_hog = hog(image_88, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)


#fig, (axes11, axes12, axes13, axes14) = pylab.subplots(1, 4, figsize=(15, 10), sharex=True, sharey=True)
#fig, (axes21, axes22, axes23, axes24) = pylab.subplots(1, 4, figsize=(15, 10), sharex=True, sharey=True)

fig, plot = pylab.subplots(3, 6)

image_8_hog_rescaled = exposure.rescale_intensity(image_8_hog, in_range=(0, 10))
image_8_original_hog_rescaled = exposure.rescale_intensity(image_8_original_hog, in_range=(0, 10))

image_67_hog_rescaled = exposure.rescale_intensity(image_67_hog, in_range=(0, 10))
image_77_hog_rescaled = exposure.rescale_intensity(image_77_hog, in_range=(0, 10))
image_88_hog_rescaled = exposure.rescale_intensity(image_88_hog, in_range=(0, 10))

plot[0, 0].set_title('Image 8')
plot[0, 0].imshow(image_8)
plot[1, 0].imshow(image_8_hog, cmap='gray')

plot[0, 1].set_title('Image 67')
plot[0, 1].imshow(image_67)
plot[1, 1].imshow(image_67_hog, cmap='gray')

plot[0, 2].set_title('Image 77')
plot[0, 2].imshow(image_77)
plot[1, 2].imshow(image_77_hog, cmap='gray')

plot[0, 3].set_title('Image 88')
plot[0, 3].imshow(image_88)
plot[1, 3].imshow(image_88_hog, cmap='gray')

image_conv = image_8 * image_67 * image_77 * image_88
fd, image_conv_hog = hog(image_conv, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
plot[0, 4].set_title('Convolution')
plot[0, 4].imshow(image_conv, cmap='gray')
plot[1, 4].imshow(image_conv_hog, cmap='gray')

image_8_hog_conv = image_8_hog * image_conv_hog
plot[2, 0].imshow(image_8_hog_conv, cmap='gray')

image_67_hog_conv = image_67_hog * image_conv_hog
plot[2, 1].imshow(image_67_hog_conv, cmap='gray')

image_77_hog_conv = image_77_hog * image_conv_hog
plot[2, 2].imshow(image_77_hog_conv, cmap='gray')

image_88_hog_conv = image_88_hog * image_conv_hog
plot[2, 3].imshow(image_88_hog_conv, cmap='gray')

plot[0, 5].set_title('Image 8 Original')
plot[0, 5].imshow(image_8_original)
plot[1, 5].imshow(image_8_original_hog, cmap='gray')

image_8_original_hog_conv = image_8_original_hog * image_conv_hog
plot[2, 5].imshow(image_8_original_hog_conv, cmap='gray')

#pylab.show()


nova_imagem = np.array([np.zeros([256, 256])], [np.zeros([256, 256])])

print(nova_imagem.shape)
#print(nova_imagem[0, 0])
#print(image_8_original[0, 0])
linha, coluna = image_8_original_hog_conv.shape

#for l in range(linha):
#    for c in range(coluna):
#        if image_8_original_hog_conv[l, c] > 0:
            #nova_imagem[l, c] = image_8_original[l, c, 0]

            #print(image_8_original[l, c])

#pylab.imshow(image_8_original)
#pylab.imshow(nova_imagem, cmap='gray')
#pylab.show()

print('FIM')