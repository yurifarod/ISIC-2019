#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:13:22 2020

@author: yurifarod
"""


import os
import cv2

#Primeiro pegaremos as imagens originais
entries = os.listdir('./Imagens/')

min_width = 450
min_height = 450
dim = (min_width, min_height)

for img in entries:
    entrada = './Imagens/' + img
    imagem = cv2.imread(entrada, cv2.IMREAD_UNCHANGED)
    
    resized = cv2.resize(imagem, dim, interpolation = cv2.INTER_AREA)
    
    h_inf = int(min_height*0.2)
    h_sup = int(min_height*0.8)
    w_inf = int(min_width*0.2)
    w_sup = int(min_width*0.8)
    roi = resized[h_inf:h_sup, w_inf:w_sup]
    
    saida = './Imagens_Roi/' + img
    
    cv2.imwrite(saida,roi)

entries = os.listdir('./Imagens_Aug/')

min_width = 450
min_height = 450
dim = (min_width, min_height)

for img in entries:
    entrada = './Imagens_Aug/' + img
    imagem = cv2.imread(entrada, cv2.IMREAD_UNCHANGED)
    
    resized = cv2.resize(imagem, dim, interpolation = cv2.INTER_AREA)
    
    h_inf = int(min_height*0.2)
    h_sup = int(min_height*0.8)
    w_inf = int(min_width*0.2)
    w_sup = int(min_width*0.8)
    roi = resized[h_inf:h_sup, w_inf:w_sup]
    
    saida = './Imagens_Roi/' + img
    
    cv2.imwrite(saida,roi)