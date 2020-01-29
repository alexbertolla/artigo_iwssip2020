import cv2
import os
import glob

#import numpy as np
#from matplotlib import pyplot as plt
#from numpy import vectorize
#pimentao_path = "pimentao.jpg"
#pimentao = cv.imread(pimentao_path)
#pimentao_redimensionado = cv.resize(pimentao, (200, 200))
#cv.imwrite('pimentao.jpg', pimentao_redimensionado)
#cv.waitKey()
#cv.destroyAllWindows()


diretorio = 'imagens/treinamento/outra_lagarta'
lista_arquivos = os.listdir(diretorio)

for arquivo in lista_arquivos:
    for file in glob.glob(diretorio + '/' + arquivo):
        #image_rgb = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        image_rgb = cv2.imread(file)
        image_rgb = cv2.resize(image_rgb, (256, 256))
        novo_nome = diretorio + '/' + '256x256_' + arquivo
        cv2.imwrite(novo_nome, image_rgb)
        print(novo_nome)


print("FIM")