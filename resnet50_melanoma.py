# -*- coding: utf-8 -*-

"""
Created on Sat Jun 30 15:30:12 2020

@author: yurifarod

"""

import numpy as np
import keras
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras import optimizers
from keras.applications.resnet50 import ResNet50
from keras.models import Model
from matplotlib import pyplot as plt

#Adicao das imagens
classe = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
classe = classe.sort_values('image')
names = classe['image']
classe_treinamento = classe.drop(columns=['image'])

train_imgs = []
for img in names:
        entrada = './Imagens_Roi/' + img
        rgb_img = plt.imread(entrada)
        train_imgs.append(rgb_img)
        


train_imgs = np.array(train_imgs)
train_imgs_scaled = train_imgs.astype('float32') 
train_imgs_scaled /= 255

previsores_treinamento = train_imgs_scaled

restnet = ResNet50(include_top = False, weights = 'imagenet', input_shape=(125,125,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, output=output)

#Aqui dizemos que a RESNET50 e treinavel
for layer in restnet.layers:
    layer.trainable = True

classificador = Sequential()
classificador.add(restnet)
classificador.add(Dense(1024, activation='relu', input_dim = (125,125,3)))
classificador.add(Dropout(0.2))
classificador.add(Dense(1024, activation='relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(9, activation='softmax'))
classificador.compile(loss='categorical_crossentropy',
                        optimizer=optimizers.RMSprop(lr=1e-5),
                        metrics=['accuracy'])

classificador.fit(previsores_treinamento,
                  classe_treinamento,
                  batch_size = 1000,
                  epochs = 100)


classificador_json = classificador.to_json()
with open('classificador.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('pesos_final.h5')