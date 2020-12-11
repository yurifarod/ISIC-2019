#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 21:13:22 2020

@author: yurifarod
"""


import os
import cv2

entries = os.listdir('./Imagens/')

min_width = 450
min_height = 576
dim = (min_width, min_height)

for img in entries:
    entrada = './Imagens_Roi/' + img
    imagem = cv2.imread(entrada, cv2.IMREAD_UNCHANGED)
    
    resized = cv2.resize(imagem, dim, interpolation = cv2.INTER_AREA)
    
    saida = './Imagens_Resized/' + img
    cv2.imwrite(saida,resized)