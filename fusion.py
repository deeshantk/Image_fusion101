import pywt
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from copy import copy
from tkinter import filedialog
import tkinter as tk


root = tk.Tk()
root.withdraw()
path = filedialog.askopenfilename()
img1 = cv2.imread(path)
path = filedialog.askopenfilename()
img2 = cv2.imread(path)
row_not_compa, col_not_compa = False, False

if img1.shape != img2.shape:
    print('Image size is different. This can lead to distortion.')
if img1[:,:,0].shape[0] % 2 != 0:
    row_not_compa = True
if img1[:,:,0].shape[1] % 2 != 0:
    col_not_compa = True
    
dispatcher = {'Max': np.maximum, 'Mean':np.mean, 'Min': np.minimum}
def imgfusion(img1, img2, ftype, wtype):
    cA_conv, rest = ftype.split('_')
    temp = pywt.dwt2(img1, 'coif5' , mode='periodization')
    cA1, (cH1, cV1, cD1) = temp
    cA2, (cH2, cV2, cD2) = pywt.dwt2(img2, 'coif5' , mode='periodization')
    r, c = cA1.shape
    
    cA = np.zeros((r, c), dtype=float)
    cH = np.zeros((r, c), dtype=float)
    cV = np.zeros((r, c), dtype=float)
    cD = np.zeros((r, c), dtype=float)
    
    for row in range(r):
        for col in range(c):
            if cA_conv == 'Mean':
                cA[row][col] = dispatcher[cA_conv]([cA1[row][col], cA2[row][col]])
            else:
                cA[row][col] = dispatcher[cA_conv](cA1[row][col], cA2[row][col])
            if rest == 'Mean':
                cH[row][col] = dispatcher[rest]([cH1[row][col], cH2[row][col]])
                cV[row][col] = dispatcher[rest]([cV1[row][col], cV2[row][col]])
                cD[row][col] = dispatcher[rest]([cD1[row][col], cD2[row][col]])
            else:
                cH[row][col] = dispatcher[rest](cH1[row][col], cH2[row][col])
                cV[row][col] = dispatcher[rest](cV1[row][col], cV2[row][col])
                cD[row][col] = dispatcher[rest](cD1[row][col], cD2[row][col])
        
    cA = np.array(cA)
    cH = np.array(cH)
    cV = np.array(cV)
    cD = np.array(cD)
    output = tuple([cH, cV, cD])
    output = tuple([cA, output])
    return pywt.idwt2(output, wtype, mode='periodization')
    
    
fusedimgR = imgfusion(img1[:,:,0], img2[:,:,0], 'Max_Max', 'coif5')
fusedimgG = imgfusion(img1[:,:,1], img2[:,:,1], 'Max_Max', 'coif5')
fusedimgB = imgfusion(img1[:,:,2], img2[:,:,2], 'Max_Max', 'coif5')
if col_not_compa:
    fusedimgR = np.delete(fusedimgR, -1, axis=1)
    fusedimgG = np.delete(fusedimgG, -1, axis=1)
    fusedimgB = np.delete(fusedimgB, -1, axis=1)
if row_not_compa:
    fusedimgR = np.delete(fusedimgR, -1, axis=0)
    fusedimgG = np.delete(fusedimgG, -1, axis=0)
    fusedimgB = np.delete(fusedimgB, -1, axis=0)
    
fused_image = copy(img2)
fused_image[:,:,0] = np.uint8(fusedimgR)
fused_image[:,:,1] = np.uint8(fusedimgG)
fused_image[:,:,2] = np.uint8(fusedimgB)
path = filedialog.askdirectory()
cv2.imwrite(path+'/fusedimage.jpg', fused_image)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(img1)
plt.title('Image 1', fontsize=30)

plt.subplot(2, 2, 2)
plt.imshow(img2)
plt.title('Image 2', fontsize=30)

plt.subplot(2, 2, 3)
plt.imshow(fused_image)
plt.title('Fused Image', fontsize=30)

plt.show()
