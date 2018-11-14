import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Data\tetris.png',0)

kernel = np.ones((5,5),np.uint8)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.imshow(gradient, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
