import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Data/mario.jpg',0)
img2 = img.copy()
template = cv2.imread('Data/mario_coin.jpg',0)
w, h = template.shape[::-1]

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

meth = methods[0]
img = img2.copy()
method = eval(meth)

# Apply template Matching
res = cv2.matchTemplate(img,template,method)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    top_left = min_loc
else:
    top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img,top_left, bottom_right, 255, 2)

plt.subplot(121),plt.imshow(res,cmap = 'gray')
plt.title('Matching Result'), plt.xticks([]), plt.yticks([])

if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    threshold = min_val*1.2
else:
    threshold = max_val*0.89

loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)

plt.subplot(122),plt.imshow(img,cmap = 'gray')
plt.title('Detected Points'), plt.xticks([]), plt.yticks([])
plt.suptitle(meth)

plt.show()
