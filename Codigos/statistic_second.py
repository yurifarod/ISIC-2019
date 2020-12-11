import numpy as np
from skimage.feature import greycomatrix, greycoprops

np.random.seed(12)
img = np.zeros((70,70),dtype=np.uint8)
for i in range(70):
    img[i] = np.random.uniform(0,256,70)

d = 1

#Para 45 graus
glcm = greycomatrix(img,[d],[45],256,symmetric=True,normed=True)

contrast_45 = greycoprops(glcm,'contrast')[0,0]
dissimilarity_45 = greycoprops(glcm,'dissimilarity')[0,0]
homogeneity_45 = greycoprops(glcm,'homogeneity')[0,0]
ASM_45 = greycoprops(glcm,'ASM')[0,0]
energy_45 = greycoprops(glcm,'energy')[0,0]
correlation_45 = greycoprops(glcm,'correlation')[0,0]

print("Resultados para 45 graus:")
print("contrast: ", contrast_45)
print("dissimilarity: ", dissimilarity_45)
print("homogeneity: ", homogeneity_45)
print("ASM: ", ASM_45)
print("energy: ", energy_45)
print("correlation: ", correlation_45)
print("\n")

#Para 90 graus
glcm = greycomatrix(img,[d],[90],256,symmetric=True,normed=True)

contrast_90 = greycoprops(glcm,'contrast')[0,0]
dissimilarity_90 = greycoprops(glcm,'dissimilarity')[0,0]
homogeneity_90 = greycoprops(glcm,'homogeneity')[0,0]
ASM_90 = greycoprops(glcm,'ASM')[0,0]
energy_90 = greycoprops(glcm,'energy')[0,0]
correlation_90 = greycoprops(glcm,'correlation')[0,0]

print("Resultados para 90 graus:")
print("contrast: ", contrast_90)
print("dissimilarity: ", dissimilarity_90)
print("homogeneity: ", homogeneity_90)
print("ASM: ", ASM_90)
print("energy: ", energy_90)
print("correlation: ", correlation_90)




    
