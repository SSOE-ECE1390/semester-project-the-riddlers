import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colormaps

img = cv2.imread("testImage.png")
B,G,R = cv2.split(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#get anything white, and make it even more white, take anything not white and make it black

th,thresh1_img = cv2.threshold(B,150,255,cv2.THRESH_BINARY)
th,thresh2_img = cv2.threshold(G,150,255,cv2.THRESH_BINARY)
th,thresh3_img = cv2.threshold(R,150,255,cv2.THRESH_BINARY)

anded = cv2.bitwise_and(thresh1_img, thresh2_img, thresh3_img)

# use contours to get the rest

contours,hierarchy = cv2.findContours(anded,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

plt.imshow(anded, cmap='gray')
plt.show()

sorted_contour = sorted(contours,key=cv2.contourArea)  # smallest to largest

mask = np.zeros([img.shape[0],img.shape[1]],dtype='uint8')

mask=cv2.drawContours(mask,sorted_contour,-1,1,-1) # The 3rd entry is the index of the contour to use, -1 means all
                                                   # the 4th entry is the color to use. SInce this is a mask, just use 1
                                                   # the 5th entry is the linewidth. -1 will fill in the contour 

im_masked = cv2.bitwise_and(img,img,mask=mask)

color = ['#641E16','b','g','r','c','m','y','#E67E22']

# Lets draw the lines around each coin
for idx in range(0,len(sorted_contour)):
    linewidth=5
    c=tuple(255*np.array(colormaps.to_rgb(color[3]))) 
    im_masked=cv2.drawContours(im_masked,sorted_contour,idx,c,linewidth)

area=np.zeros((len(sorted_contour)))
for i in range(0,len(sorted_contour)):
    area[i]=cv2.contourArea(sorted_contour[i])

plt.imshow(im_masked)
plt.show()

# next steps are to compute a distance based on corner features


plt.imshow(img_gray, cmap='gray')
plt.show()
