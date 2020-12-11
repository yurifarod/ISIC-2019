#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 21:28:27 2019

@author: yurifarod

Colocar as novas caracterisicas no csv e ver se esta tudo batendo
"""
    import os
import csv
import math
import cv2
import numpy as np
from scipy.stats import entropy, skew, kurtosis
from matplotlib import pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage.color import rgb2hsv

def rgb2gray(img):
	return np.dot(img[:,:,:3],[.299,.587,.144]).astype('float')

def rgb2blue(img):
    return img[:,:,0]

def rgb2green(img):
    return img[:,:,1]

def rgb2red(img):
    return img[:,:,2]

entries = os.listdir('./Imagens_Roi/')


with open('dados_img.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
    spamwriter.writerow(['nm_img', 'gss_media', 'gss_variancia', 'gss_mediana', 'gss_quantile', 'gss_2quantile', 'gss_coef',
                         'rs_media', 'rs_variancia', 'rs_mediana', 'rs_quantile', 'rs_2quantile', 'rs_coef',
                         'gs_media', 'gs_variancia', 'gs_mediana', 'gs_quantile', 'gs_2quantile', 'gs_coef',
                         'bs_media', 'bs_variancia', 'bs_mediana', 'bs_quantile', 'bs_2quantile', 'bs_coef',
                         'contrast_45', 'dissimilarity_45', 'homogeneity_45', 'ASM_45', 'energy_45', 'correlation_45',
                         'contrast_90', 'dissimilarity_90', 'homogeneity_90', 'ASM_90', 'energy_90', 'correlation_90',
                         'gss_entropy', 'gss_skew', 'gss_kurtosis', 'gss_std',
                         'rs_entropy', 'rs_skew', 'rs_kurtosis', 'rs_std',
                         'gs_entropy', 'gs_skew', 'gs_kurtosis', 'gs_std',
                         'bs_entropy', 'bs_skew', 'bs_kurtosis', 'bs_std',
                         'hue_media', 'hue_variancia', 'hue_mediana', 'hue_entropy', 'hue_skew', 'hue_kurtosis', 'hue_std',
                         'sat_media', 'sat_variancia', 'sat_mediana', 'sat_entropy', 'sat_skew', 'sat_kurtosis', 'sat_std',
                         'value_media', 'value_variancia', 'value_mediana', 'value_entropy', 'value_skew', 'value_kurtosis', 'value_std',
                         'perimeter', 'area', 'cx', 'cy', 'circle_x', 'circle_y', 'circle_area', 'circle_perimeter', 'eq_diameter',
                         'compac', 'major_axis', 'minor_axis', 'aspect', 'elongation', 'rectangularity',
                         'dft_media', 'dft_variancia', 'dft_mediana', 'dft_quantile', 'dft_2quantile', 'dft_kurtosis', 'dft_std',
                         'mag_media', 'mag_variancia', 'mag_mediana', 'mag_quantile', 'mag_2quantile',
                         'mag_entropy', 'mag_skew', 'mag_kurtosis', 'mag_std'])

    yum = 0
    for img in entries:
        entrada = './Imagens_Roi/' + img
        
        rgb_img = plt.imread(entrada)

        x, y, cor = rgb_img.shape
        
        
        gs_img = rgb2gray(rgb_img)
        rs_img = rgb2red(rgb_img)
        gs_img = rgb2green(rgb_img)
        bs_img = rgb2blue(rgb_img)
        
        #Estatistica de segunda ordem
        glcm = greycomatrix(gs_img,[1],[45],256,symmetric=True,normed=True)
        contrast_45 = greycoprops(glcm,'contrast')[0,0]
        dissimilarity_45 = greycoprops(glcm,'dissimilarity')[0,0]
        homogeneity_45 = greycoprops(glcm,'homogeneity')[0,0]
        ASM_45 = greycoprops(glcm,'ASM')[0,0]
        energy_45 = greycoprops(glcm,'energy')[0,0]
        correlation_45 = greycoprops(glcm,'correlation')[0,0]
        
        glcm = greycomatrix(gs_img,[1],[90],256,symmetric=True,normed=True)

        contrast_90 = greycoprops(glcm,'contrast')[0,0]
        dissimilarity_90 = greycoprops(glcm,'dissimilarity')[0,0]
        homogeneity_90 = greycoprops(glcm,'homogeneity')[0,0]
        ASM_90 = greycoprops(glcm,'ASM')[0,0]
        energy_90 = greycoprops(glcm,'energy')[0,0]
        correlation_90 = greycoprops(glcm,'correlation')[0,0]
        
        
        
        #Teremos 6 entradas pra gray scale e para cada canal de cor
        gss_media = np.mean(gs_img)
        gss_variancia = np.var(gs_img)
        gss_mediana = np.median(gs_img)
        gss_quantile = np.quantile(gs_img, q=0.25)
        gss_2quantile = np.quantile(gs_img, q=0.75)
        gss_coef = np.corrcoef(gs_img).mean()
        if math.isnan(gss_coef):
                gss_coef = 0
        
        #Teremos 6 entradas pra R
        rs_media = np.mean(rs_img)
        rs_variancia = np.var(rs_img)
        rs_mediana = np.median(rs_img)
        rs_quantile = np.quantile(rs_img, q=0.25)
        rs_2quantile = np.quantile(rs_img, q=0.75)
        rs_coef = np.corrcoef(gs_img, rs_img).mean()
        if math.isnan(rs_coef):
                rs_coef = 0
        
        #Teremos 6 entradas pra G
        gs_media = np.mean(gs_img)
        gs_variancia = np.var(gs_img)
        gs_mediana = np.median(gs_img)
        gs_quantile = np.quantile(gs_img, q=0.25)
        gs_2quantile = np.quantile(gs_img, q=0.75)
        gs_coef = np.corrcoef(gs_img, gs_img).mean()
        if math.isnan(gs_coef):
                gs_coef = 0
        
        #Teremos 6 entradas pra B
        bs_media = np.mean(bs_img)
        bs_variancia = np.var(bs_img)
        bs_mediana = np.median(bs_img)
        bs_quantile = np.quantile(bs_img, q=0.25)
        bs_2quantile = np.quantile(bs_img, q=0.75)
        bs_coef = np.corrcoef(gs_img, bs_img).mean()
        if math.isnan(bs_coef):
                bs_coef = 0
        
        #EXTRACAO DE FEATURES DO ARTIGO!
        
        #First Order - GreySacle
        gss_entropy = np.mean(entropy(gs_img))
        gss_skew = np.mean(skew(gs_img))
        gss_kurtosis = np.mean(kurtosis(gs_img))
        gss_std = np.std(gs_img)
        
        #First Order - R
        rs_entropy = np.mean(entropy(rs_img))
        rs_skew = np.mean(skew(rs_img))
        rs_kurtosis = np.mean(kurtosis(rs_img))
        rs_std = np.std(rs_img)
        
        #First Order - G
        gs_entropy = np.mean(entropy(gs_img))
        gs_skew = np.mean(skew(gs_img))
        gs_kurtosis = np.mean(kurtosis(gs_img))
        gs_std = np.std(gs_img)
        
        #First Order - B
        bs_entropy = np.mean(entropy(bs_img))
        bs_skew = np.mean(skew(bs_img))
        bs_kurtosis = np.mean(kurtosis(bs_img))
        bs_std = np.std(bs_img)
        
        hsv_img = rgb2hsv(rgb_img)
        
        hue_img = hsv_img[:, :, 0]
        
        hue_media = np.mean(hue_img)
        hue_variancia = np.var(hue_img)
        hue_mediana = np.median(hue_img)
        hue_entropy = np.mean(entropy(hue_img))
        if math.isnan(hue_entropy):
                hue_entropy = 0
        hue_skew = np.mean(skew(hue_img))
        hue_kurtosis = np.mean(kurtosis(hue_img))
        hue_std = np.std(hue_img)
        
        sat_img = hsv_img[:, :, 1]
        sat_media = np.mean(sat_img)
        sat_variancia = np.var(sat_img)
        sat_mediana = np.median(sat_img)
        sat_entropy = np.mean(entropy(sat_img))
        if math.isnan(sat_entropy):
                sat_entropy = 0
        sat_skew = np.mean(skew(sat_img))
        sat_kurtosis = np.mean(kurtosis(sat_img))
        sat_std = np.std(sat_img)
        
        value_img = hsv_img[:, :, 2]
        value_media = np.mean(value_img)
        value_variancia = np.var(value_img)
        value_mediana = np.median(value_img)
        value_entropy = np.mean(entropy(value_img))
        if math.isnan(value_entropy):
                value_entropy = 0
        value_skew = np.mean(skew(value_img))
        value_kurtosis = np.mean(kurtosis(value_img))
        value_std = np.std(value_img)
        
        cv2_img = cv2.imread(entrada, 0)
        ret, thresh = cv2.threshold(cv2_img,127,255,0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        
        if len(contours) < 1:
            #Nao conseguiu tracar um cotorno
            perimeter = 0
            area = 0
            cx = 0
            cy = 0
            circle_x = 225
            circle_y = 225
            circle_area = 0
            circle_perimiter = 0
            eq_diameter = 0
            compac = 0
            major_axis = 0
            minor_axis = 0
            aspect = 0
            elongation = 0
            rectangularity = 0
            
        else:
            cnt = contours[0]
            
            M = cv2.moments(cnt)
            perimeter = cv2.arcLength(cnt,True)
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            if M['m00'] == 0:
                cx = 0
                cy = 0
            else:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            
            (x,y),raio = cv2.minEnclosingCircle(cnt)
            circle_x = x
            circle_y = y
            #Lembrar de colocar o raio
            if circle_x == 0:
                circle_x = 225
            if circle_y == 0:
                circle_y = 255
            circle_area = 2*math.pi*raio
            circle_perimiter = math.pi*raio*raio
            
            eq_diameter = math.sqrt(4*area/math.pi)
            compac = 4*math.pi*area/(circle_perimiter*circle_perimiter)
            
            major_axis = circle_x
            minor_axis = circle_y
            if circle_x < circle_y:
                major_axis = circle_y
                minor_axis = circle_x
            
            aspect = minor_axis/major_axis
            elongation = 1 - aspect
            rectangularity = area/(major_axis * minor_axis)
        
        dft = cv2.dft(np.float32(cv2_img),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        
        #Isso aqui vai mudar, vai tirar igual Elwislan
        
        dft_media = np.mean(dft)
        dft_variancia = np.var(dft)
        dft_mediana = np.median(dft)
        dft_quantile = np.quantile(dft, q=0.25)
        dft_2quantile = np.quantile(dft, q=0.75)
        dft_entropy = np.mean(entropy(dft))
        dft_skew = np.mean(skew(dft))
        dft_kurtosis = np.mean(kurtosis(dft))
        dft_std = np.std(dft)
        
        mag_media = np.mean(magnitude_spectrum)
        mag_variancia = np.var(magnitude_spectrum)
        mag_mediana = np.median(magnitude_spectrum)
        mag_quantile = np.quantile(magnitude_spectrum, q=0.25)
        mag_2quantile = np.quantile(magnitude_spectrum, q=0.75)
        mag_entropy = float(np.mean(entropy(magnitude_spectrum)))
        mag_skew = np.mean(skew(magnitude_spectrum))
        mag_kurtosis = np.mean(kurtosis(magnitude_spectrum))
        mag_std = np.std(magnitude_spectrum)
        
        if (mag_media == math.inf*-1) or (mag_media == math.inf) or math.isnan(mag_media):
                mag_media = 0
        if (mag_variancia == math.inf*-1) or (mag_variancia == math.inf) or math.isnan(mag_variancia):
                mag_variancia = 0
        if (mag_mediana == math.inf*-1) or (mag_mediana == math.inf) or math.isnan(mag_mediana):
                mag_mediana = 0
        if (mag_quantile == math.inf*-1) or (mag_quantile == math.inf) or math.isnan(mag_quantile):
                mag_quantile = 0
        if (mag_2quantile == math.inf*-1) or (mag_2quantile == math.inf) or math.isnan(mag_2quantile):
                mag_2quantile = 0
        if (mag_entropy == math.inf*-1) or (mag_entropy == math.inf) or math.isnan(mag_entropy):
                mag_entropy = 0
        if (mag_skew == math.inf*-1) or (mag_skew == math.inf) or math.isnan(mag_skew):
                mag_skew = 0
        if (mag_kurtosis == math.inf*-1) or (mag_kurtosis == math.inf) or math.isnan(mag_kurtosis):
                mag_kurtosis = 0
        if (mag_std == math.inf*-1) or (mag_std == math.inf) or math.isnan(mag_std):
                mag_std = 0
        #Escrevendo no arquivo
        spamwriter.writerow([img, gss_media, gss_variancia, gss_mediana, gss_quantile, gss_2quantile, gss_coef,
                              rs_media, rs_variancia, rs_mediana, rs_quantile, rs_2quantile, rs_coef,
                              gs_media, gs_variancia, gs_mediana, gs_quantile, gs_2quantile, gs_coef,
                              bs_media, bs_variancia, bs_mediana, bs_quantile, bs_2quantile, bs_coef,
                              contrast_45, dissimilarity_45, homogeneity_45, ASM_45, energy_45, correlation_45,
                              contrast_90, dissimilarity_90, homogeneity_90, ASM_90, energy_90, correlation_90,
                              gss_entropy, gss_skew, gss_kurtosis, gss_std,
                              rs_entropy, rs_skew, rs_kurtosis, rs_std,
                              gs_entropy, gs_skew, gs_kurtosis, gs_std,
                              bs_entropy, bs_skew, bs_kurtosis, bs_std,
                              hue_media, hue_variancia, hue_mediana, hue_entropy, hue_skew, hue_kurtosis, hue_std,
                              sat_media, sat_variancia, sat_mediana, sat_entropy, sat_skew, sat_kurtosis, sat_std,
                              value_media, value_variancia, value_mediana, value_entropy, value_skew, value_kurtosis, value_std,
                              perimeter, area, cx, cy, circle_x, circle_y, circle_area, circle_perimiter, eq_diameter,
                              compac, major_axis, minor_axis, aspect, elongation, rectangularity,
                              dft_media, dft_variancia, dft_mediana, dft_quantile, dft_2quantile, dft_kurtosis, dft_std,
                              mag_media, mag_variancia, mag_mediana, mag_quantile, mag_2quantile,
                              mag_entropy, mag_skew, mag_kurtosis, mag_std])
