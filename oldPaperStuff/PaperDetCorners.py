import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist


#function to order corner points in a clockwise manner for drawing the shape
def order_points(pts):

    xSort = pts[np.argsort(pts[:,0]),:]

    leftMost = xSort[:2,:]
    rightMost = xSort[2:,:]

    leftMost = leftMost[np.argsort(leftMost[:,1]),:]
    (tl,bl) = leftMost

    D = dist.cdist(tl[np.newaxis],rightMost, "euclidean")[0]
    (br,tr) = rightMost[np.argsort(D)[::-1],:]

    return np.array([tl,tr,br,bl],dtype="float32")

def paper_corners(img):

        #image preprocessing 
    img = cv2.imread('testImage.png') 

    #find image width and height
    w,h = img.shape[:2]
    B,G,R = cv2.split(img)

    #convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    th,thresh1_img = cv2.threshold(B,170,255,cv2.THRESH_BINARY)
    th,thresh2_img = cv2.threshold(G,170,255,cv2.THRESH_BINARY)
    th,thresh3_img = cv2.threshold(R,170,255,cv2.THRESH_BINARY)

    anded = cv2.bitwise_and(thresh1_img, thresh2_img, thresh3_img)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(anded,kernel,iterations=8) 
    dilation = cv2.dilate(erosion,kernel,iterations=15)

    #blur image, larger kernel provides closer fit than smaller kernel, but too big a kernel shrinks the paper
    blur = cv2.GaussianBlur(dilation,(55,55),0)

    #threshold image for white
    thresh = cv2.threshold(blur,25,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    #morphological transformation
    kernel = np.ones((7,7),np.uint8)
    morph = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel=kernel)
    morph = cv2.morphologyEx(morph,cv2.MORPH_OPEN, kernel=kernel)

    #find contours, outlines of blurred ROIs 
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    corners = cv2.goodFeaturesToTrack(thresh,4,0.01,500) #identify corners in image
    corners = np.int0(corners)

    for i in corners:

        x,y = i.ravel()
        cv2.circle(img,(x,y),3,255,-1) #draw circle at each corner

    corners_2d = corners.reshape(-1,2)
    corners_ord = order_points(corners_2d) #put corners in clockwise order

    corners_ord = np.int0(corners_ord)

    paper = np.zeros_like(img)

    #draw contour lines on the image using coordinates found in largest contour area
    cv2.drawContours(paper,[corners_ord], 0, (255,255,255),-1) 

    peri = cv2.arcLength(corners_ord,True) #parameter for approximation accuracy
    #approximate the paper outline with fewer indices, i.e. simplify a more complex shape to rectangle indices
    corner = cv2.approxPolyDP(corners_ord,0.04*peri, True)

    x,y,w,h = cv2.boundingRect(corners_ord)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    polygon = img.copy()

    #overlay the new shape/polygon
    cv2.polylines(polygon,[corner],True,(255,0,0),5,cv2.LINE_AA)

    return polygon