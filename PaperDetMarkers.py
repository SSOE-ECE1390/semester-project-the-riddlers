import cv2
from cv2 import aruco
import os
import numpy as np
import matplotlib.pyplot as plt

from packaging import version  # Installed with setuptools, so should already be installed in your env.

def paper_markers(img):

    #depending on version of cv2, carry out aruco marker detection
    if version.parse(cv2.__version__) >= version.parse("4.7.0"): 
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        detectorParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detectorParams) #detect markers on paper
        corners, ids, rejected = detector.detectMarkers(img)
    else:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
        detectorParams = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, dictionary, parameters=detectorParams #detect markers on paper
        )

    polygon_pts = []

    for i in range(len(ids)):

        marker_corners = corners[i][0] #extracts corners of each marker

        for corner in np.nditer(marker_corners):

            polygon_pts.append(corner) #append marker corners to an array

    polygon = np.array(polygon_pts,np.int32).reshape((-1,1,2))

    x,y,w,h = cv2.boundingRect(polygon) #find rectangle that bounds the contour
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #draw rectangle around markers

    rect2 = cv2.minAreaRect(polygon) #find smallest possible rectangle to bind contour
    box = cv2.boxPoints(rect2)
    box = np.intp(box)
    img = cv2.drawContours(img,[box],0,(0,0,255),2) #draw rectangle around paper

    roi = img[y:y+h,x:x+w] #extract ROI of just the paper

    return roi, (x,y,w,h)

if __name__=='__main__':
    img = cv2.imread('PaperMarkers2.jpg')
    plt.imshow(paper_markers(img))
    plt.show()

