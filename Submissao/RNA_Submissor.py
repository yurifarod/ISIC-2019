#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:21:33 2020

@author: yurifarod
"""
import csv
import pandas as pd
import statistics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json


dados_imagem = pd.read_csv('dados_img.csv')
dados_fornecidos = pd.read_csv('ISIC_2019_Test_Metadata.csv')

dados_imagem = dados_imagem.sort_values('nm_img')
dados_fornecidos = dados_fornecidos.sort_values('image')

idade = dados_fornecidos.iloc[:, 1].values
local = dados_fornecidos.iloc[:, 2].values
sexo = dados_fornecidos.iloc[:, 3].values

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
dados_imagem['local9'] = 0

nome_imagem = dados_imagem['nm_img']
dados_imagem = dados_imagem.drop(columns=['nm_img'])
previsores = dados_imagem

normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

#Importar aqui a rede neural e os pesos
arquivo = open('classificador.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

classificador = model_from_json(estrutura_rede)
classificador.load_weights('pesos_final.h5')


previsao = classificador.predict(previsores)
# previsao = (previsao > 0.5)

count = 0

with open('sub_archive.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['image','MEL','NV','BCC','AK','BKL','DF','VASC','SCC','UNK'])
    for i in previsao:
        M = "{:.10f}".format(i[0])
        N = "{:.10f}".format(i[1])
        B = "{:.10f}".format(i[2])
        A = "{:.10f}".format(i[3])
        BK = "{:.10f}".format(i[4])
        D = "{:.10f}".format(i[5])
        V = "{:.10f}".format(i[6])
        S = "{:.10f}".format(i[7])
        U = "{:.10f}".format(i[8])
        # M = i[0]
        # N = i[1]
        # B = i[2]
        # A = i[3]
        # BK = i[4]
        # D = i[5]
        # V = i[6]
        # S = i[7]
        # U = i[8]
        spamwriter.writerow([nome_imagem[count][0:12], M, N, B, A, BK, D, V, S, U])
        count += 1

#Olhar o NAN
h = 3887
M = "{:.10f}".format(previsao[h][0])
N = "{:.10f}".format(previsao[h][1])
B = "{:.10f}".format(previsao[h][2])
A = "{:.10f}".format(previsao[h][3])
BK = "{:.10f}".format(previsao[h][4])
D = "{:.10f}".format(previsao[h][5])
V = "{:.10f}".format(previsao[h][6])
S = "{:.10f}".format(previsao[h][7])
U = "{:.10f}".format(previsao[h][8])
print(M, N, B, A, BK, D, V, S, U)