#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Sat Jun 29 15:30:12 2019

@author: yurifarod

"""

import statistics
import pandas as pd
import numpy as np
import keras.metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

dados_imagem = pd.read_csv('dados_img.csv')
dados_fornecidos = pd.read_csv('ISIC_2019_Training_Metadata.csv')

dados_imagem = dados_imagem.sort_values('nm_img')
dados_fornecidos = dados_fornecidos.sort_values('image')

idade = dados_fornecidos.iloc[:, 1].values
local = dados_fornecidos.iloc[:, 2].values
sexo = dados_fornecidos.iloc[:, 4].values

labelencoder = LabelEncoder()

sexo = labelencoder.fit_transform(sexo.astype(str))
local = labelencoder.fit_transform(local.astype(str))

idade_mediana = statistics.median(idade)
idade[np.isnan(idade)] = int(idade_mediana)

onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
sexo = sexo.reshape(len(sexo), 1)
sexo = onehot_encoder.fit_transform(sexo)

local = local.reshape(len(local), 1)
local = onehot_encoder.fit_transform(local)

dados_imagem['idade'] = idade
dados_imagem['sexo1'] = sexo[:, 0]
dados_imagem['sexo2'] = sexo[:, 1]
dados_imagem['sexo3'] = sexo[:, 2]
dados_imagem['local1'] = local[:, 0]
dados_imagem['local2'] = local[:, 1]
dados_imagem['local3'] = local[:, 2]
dados_imagem['local4'] = local[:, 3]
dados_imagem['local5'] = local[:, 4]
dados_imagem['local6'] = local[:, 5]
dados_imagem['local7'] = local[:, 6]
dados_imagem['local8'] = local[:, 7]
dados_imagem['local9'] = local[:, 8]

dados_imagem = dados_imagem.drop(columns=['nm_img'])
previsores = dados_imagem

normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

classe = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
classe = classe.sort_values('image')
classe = classe.drop(columns='image')

previsores_treinamento = previsores
classe_treinamento = classe

contador = classe_treinamento.sum(0)

classificador = Sequential()
classificador.add(Dense(units = 1024,
                        activation = 'relu',
                        kernel_initializer = 'random_uniform',
                        input_dim = 117))

classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1024,
                        kernel_initializer = 'random_uniform',
                        activation = 'relu'))

# Na camada de saida a funcao softmax e necessaria qnd se ha mais de uma classe
# Ha tb uma saida para cada classe
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 9,
                        kernel_initializer = 'random_uniform',
                        activation = 'softmax'))

# Metrica loss e metrica tb alteradas pelas multiplas opcoes
classificador.compile(optimizer = 'adam',
                      loss = 'categorical_crossentropy',
                      metrics=['accuracy'])

#Aqui o treinamento
#Ver taxa de aprendizagem no padrao Elwislan (0.00001)
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 500,
                  epochs = 300)

classificador_json = classificador.to_json()
with open('classificador.json', 'w') as json_file:
    json_file.write(classificador_json)

classificador.save_weights('pesos_final.h5')