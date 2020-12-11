#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:57:17 2020

@author: yurifarod
"""


import os
import cv2

entries = os.listdir('./Imagens/')

min_width = 99999999
min_height = 99999999

for img in entries:
    entrada = './Imagens/' + img
    img = cv2.imread(entrada, cv2.IMREAD_UNCHANGED)
    width = img.shape[0]
    height = img.shape[1]
    
    if min_width > width:
        min_width = width
    if min_height > height:
        min_height = height
    
dim = (min_width, min_height)

# resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)