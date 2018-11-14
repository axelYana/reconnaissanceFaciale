import cv2
import numpy as np
from matplotlib import pyplot as plt


def proximity_cut(x, ref, thd):
    """Returns a float between 0 and 1 characterising the proximity between a number x
    and a reference ref, being 0 if over a certain threshold thd
    :param x: float
    :param ref: float
    :param thd: float
    :return: float between 0 and 1
    """
    if abs(x-ref) >= thd:
        return 0
    else:
        return 1-abs(x-ref)/thd


def drawLine(img, center, length, angle, angleInDegrees=True):
    """
    :param img: an openCv image
    :param center: (1,2) array
    :param length: float
    :param angle: float
    :param angleInDegrees: bool
    :return: None
    """
    if angleInDegrees:
        angle *= (2*np.pi)/360
    x1 = int(center[0] - length/2 * np.cos(angle))
    x2 = int(center[0] + length/2 * np.cos(angle))
    y1 = int(center[1] - length/2 * np.sin(angle))
    y2 = int(center[1] + length/2 * np.sin(angle))
    cv2.line(img, (x1, x2), (y1, y2), (255, 255, 255))


ori_img = cv2.imread('Data/bouder.jpg',0)
img = ori_img.copy()

size = np.shape(img)
print(size)
img = cv2.resize(img, (128, 128))
print(np.shape(img))

# Calcul les coordonnées du gradient selon x et y
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)

# Calcul des angles et magnitudes du gradient
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)


plt.subplot(2,3,1),plt.imshow(ori_img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(img,cmap = 'gray')
plt.title('Squared'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(mag,cmap = 'gray')
plt.title('Gradient'), plt.xticks([]), plt.yticks([])


# Calcul de l'histogramme local sur un carré de 8x8 pixels
hist = np.zeros([16, 16, 8])
HoG = np.zeros([128, 128])
for p in range(16):
    for q in range(16):
        for i in range(8):
            for j in range(8):
                for ang_i in range(8):
                    hist[p,q,ang_i] += mag[p*8+i,q*8+j]*proximity_cut(angle[p*8+i,q*8+j], ang_i*20, 20)
        for ang_i in range(8):
            drawLine(HoG, [p*8+4, q*8+4], hist[p,q,ang_i], ang_i*20, angleInDegrees=True)
print(hist)


plt.subplot(2,3,5),plt.imshow(HoG,cmap = 'gray')
plt.title('Hog representation'), plt.xticks([]), plt.yticks([])


plt.show()

