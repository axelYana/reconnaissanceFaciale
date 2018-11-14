import cv2
import numpy as np
from matplotlib import pyplot as plt

#charger une image
img = cv2.imread('C:/Users/Axel/Pictures/Saved Pictures/profil.jpg')
rows,cols,ch = img.shape


#Choisir un système de points (pts1) qui vont être translatté (en pts2)
pts1 = np.float32([[353,505],[610,494],[493,809]])
pts2 = np.float32([[200,750],[1200,750],[650,1500]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()