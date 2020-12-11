#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 15:30:12 2019

@author: yurifarod
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np
import statistics

dados_imagem = pd.read_csv('dados_img.csv')
dados_fornecidos = pd.read_csv('ISIC_2019_Training_Metadata.csv')
idade = dados_fornecidos.iloc[:, 1].values
local = dados_fornecidos.iloc[:, 2].values
sexo = dados_fornecidos.iloc[:, 4].values

labelencoder = LabelEncoder()
sexo = labelencoder.fit_transform(sexo)
local = labelencoder.fit_transform(local)

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

previsores = dados_imagem

classe = pd.read_csv('ISIC_2019_Training_GroundTruth.csv').iloc[:, 1:9]

classe_v = classe.values
            
classe_teste = [np.argmax(t) for t in classe_v]

def criar_rede(optimizer, kernel_initializer, activation, neurons, perda):
    
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation,
                            kernel_initializer = kernel_initializer,
                            input_dim = 49))
    classificador.add(Dense(units = neurons, activation = activation))
    
    classificador.add(Dense(units = 8, activation = activation))
    
    classificador.compile(optimizer = optimizer, loss = perda,
                          metrics = ['accuracy'])
    return classificador
    
classificador = KerasClassifier(build_fn = criar_rede)
parametros = {'batch_size': [120],
              'epochs': [100],
              'optimizer': ['sgd', 'adamax', 'adam'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh', 'sigmoid', 'softmax'],
              'neurons': [25, 27],
              'perda': ['sparse_categorical_crossentropy', 'categorical_crossentropy']}

grid_search = GridSearchCV(estimator = classificador,
                           param_grid = parametros,
                           scoring = 'accuracy',
                           cv = 2,
                           verbose = True)

grid_search = grid_search.fit(previsores, classe_teste)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_