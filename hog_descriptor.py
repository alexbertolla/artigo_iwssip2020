from matplotlib import pylab as pylab
from skimage.io import imread
from skimage import exposure
from skimage.color import rgb2gray, rgb2hsv, rgb2ycbcr, hsv2rgb, ycbcr2rgb, gray2rgb
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
image2 = imread('imagens/imagem_88.jpg')

image_gray = rgb2gray(imread('imagens/imagem_77_seg.jpg'))
fd_gray, image_hog_gray = hog(image_gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)


fd, image_hog = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
fd2, image2_hog2 = hog(image2, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)

#print(image.shape)
linha, coluna, dimension = image.shape

image_red = np.zeros([256, 256])
image_green = np.zeros([256, 256])
image_blue = np.zeros([256, 256])
nova_imagem = np.zeros([256, 256])
for l in range(linha):
    for c in range(coluna):
        image_red[l, c] = int(image[l, c, 0])
        image_green[l, c] = int(image[l, c, 1])
        image_blue[l, c] = int(image[l, c, 2])
        r = image_red[l, c]
        pixel_c = int((0.2125 * int(r)) + (0.7154 * image_green[l, c]) + (0.0721 * image_blue[l, c]))
        nova_imagem[l, c] = pixel_c
        if pixel_c < 255:

            if image_hog[l, c] > 0:
                print('nova_imagem = ', nova_imagem[l, c])
                print('image_hog = ', image_hog[l, c])
                print('image_hog_gray = ', image_hog_gray[l, c])
                print('image_gray = ', image_gray[l, c])



#print(image_red)
#pixel_max = 0
#pixel_min = 0
#nova_imagem = np.zeros([256, 256])


#limiar >=0.898 <= 0.996
#for l in range(linha):
#    for c in range(coluna):
#        print(image[l, c])
#        nova_imagem[l, c] = 1.0
#        print(hog_image[l, c])
#        if image[l, c] >= 0.898 and image[l, c] <= 0.997:
#            print(image[l, c])
#            nova_imagem[l, c] = image[l, c]
#        else:
#            image2[l, c] = 128

fig, (axes1, axes2, axes3) = pylab.subplots(1, 3, figsize=(15, 10), sharex=True, sharey=True)
image_hog_rescaled = exposure.rescale_intensity(image_hog, in_range=(0, 10))
image_hog_rescaled2 = exposure.rescale_intensity(image2_hog2, in_range=(0, 10))

axes1.axis('off'), axes1.imshow(image), axes1.set_title('Imput Image')
axes2.axis('off'), axes2.imshow(image2), axes2.set_title('Histogram of Oriented Gradients')

img_conv = exposure.rescale_intensity(image2_hog2 * image_hog, in_range=(0, 10))
axes3.axis('off'), axes3.imshow(nova_imagem, cmap='gray'), axes3.set_title('TESTE')


#pylab.figure(figsize=(10, 5))
#pylab.axis('off')
#pylab.imshow(nova_imagem, cmap='gray')

#print(gray2rgb(nova_imagem))

#pylab.hist(image_red.ravel())
#axes1.imshow(image_red, cmap='gray')
#pylab.plot()
#pylab.show()

#pylab.hist(image_green.ravel())
#axes2.imshow(image_green, cmap='gray')
#pylab.plot()
#pylab.show()

#pylab.hist(image_blue.ravel())
#axes3.imshow(image_blue, cmap='gray')
#pylab.plot()



#pylab.show()


print('FIM')