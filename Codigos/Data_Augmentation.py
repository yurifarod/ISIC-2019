#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:11:15 2020

@author: yurifarod

Este codigo tem como intuito efetuar a Data Augmentation do projeto!

"""

import os
import cv2
import random
import csv
import numpy as np
import pandas as pd
from skimage.util import random_noise

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

classe = pd.read_csv('ISIC_2019_Training_GroundTruth.csv').iloc[:, 1:9]
linhas_classe = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
linhas_meta = pd.read_csv('ISIC_2019_Training_Metadata.csv')

classes_existentes = ['0', '1', '2', '3', '4', '5', '6', '7']

max = 0
classe_max = 'Vazio'
qtd_class = []

for i in classes_existentes:
    qtd = classe.groupby(i).size()[1]
    qtd_class.append(qtd)

    if qtd > max:
        max = qtd
        classe_max = i

#Todos as classes deveram ter a msm Quantidade da maior classe!
#Agora iniciamos a leitura das imagens e o aumento dos dados

address = './Imagens/'
entries = os.listdir(address)
fimExec = True
count = 0
while fimExec:
    index = random.randint(1, 25330)
    lesao_classe = int(pd.DataFrame(classe.loc[index]).idxmax())
    img = entries[index]
    entrada = './Imagens/' + img
    imagem = cv2.imread(entrada, cv2.IMREAD_UNCHANGED)
    alterada = imagem

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 30)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    #Com a iamgem rotacionada aplicamos o metodo Gaussian Blur
    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    #Com a iamgem rotacionada aplicamos o metodo Salt and Peper
    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    #Agora faremos mais rotacoes e o mesmo procedimento para cada uma dessas rotacoes
    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 60)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 90)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 120)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 150)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 180)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 210)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 240)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 270)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 300)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada = rotate(imagem, 330)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Gauss = cv2.GaussianBlur(alterada,(5,5),cv2.BORDER_DEFAULT)

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Gauss)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    if qtd_class[lesao_classe] < max:
        count += 1

        alterada_Salt = random_noise(alterada, mode='s&p',amount=0.3)
        alterada_Salt = np.array(255*alterada_Salt, dtype = 'uint8')

        new_img = str(count) +"_" + img
        saida = './Imagens_Aug/' + new_img
        cv2.imwrite(saida,alterada_Salt)

        with open('ISIC_2019_Training_GroundTruth.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lc_edit = linhas_classe.loc[index]
            lc_edit['image'] = new_img
            spamwriter.writerow(lc_edit)

        with open('ISIC_2019_Training_Metadata.csv','a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            lm_edit = linhas_meta.loc[index]
            lm_edit['image'] = new_img
            spamwriter.writerow(lm_edit)

        qtd_class[lesao_classe] += 1

    #Aqui aplicamos as transformacoes para a imagem rotacionada em varios angulos
    fimExec = False
    for i in range(8):
        if qtd_class[i] < max:
            fimExec = True

print('Quantidade por classes: ')
for i in range(8):
    print('Classe: '+str(i))
    print('Quantidade: '+str(qtd_class[i]))
    print('#################################')
