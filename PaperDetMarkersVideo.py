import cv2
from cv2 import aruco
import os
import numpy as np
import matplotlib.pyplot as plt
import PaperDetMarkers

from packaging import version  # Installed with setuptools, so should already be installed in your env.

def paper_markers(img):

    #img = cv2.imread(img)
    marker_detected = False

    if version.parse(cv2.__version__) >= version.parse("4.7.0"):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        detectorParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
        corners, ids, rejected = detector.detectMarkers(img)
    else:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
        detectorParams = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(
            img, dictionary, parameters=detectorParams
        )

    if ids is not None:
        marker_detected = True

        polygon_pts = []

        if len(ids) == 4: 

            for i in range(len(ids)):

                marker_corners = corners[i][0]

                for corner in np.nditer(marker_corners):

                    polygon_pts.append(corner)

            polygon = np.array(polygon_pts,np.int32).reshape((-1,1,2))

            x,y,w,h = cv2.boundingRect(polygon) #find rectangle that bounds the contour
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            rect2 = cv2.minAreaRect(polygon) #find smallest possible rectangle to bind contour
            box = cv2.boxPoints(rect2)
            box = np.int0(box)
            img = cv2.drawContours(img,[box],0,(0,0,255),2)

            roi = img[y:y+h,x:x+w]

        else:
           marker_detected = False
           roi = img 

    else:
        marker_detected = False
        roi = img

    return marker_detected,roi


if __name__=='__main__':
    img = cv2.imread('frame')
    plt.imshow(paper_markers(img))
    plt.show()

#win_name = "Camera"
#cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#result = None
#source = cv2.VideoCapture(0)

#alive = True

#while alive:
    #has_frame, frame = source.read()
    #if not has_frame:
        #break

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #marker_detected,roi = paper_markers(frame)

    #if marker_detected is True:
        #print("Paper Detected!")

    #cv2.imshow(win_name, frame)

    #key = cv2.waitKey(1)
    #if key == ord("Q") or key == ord("q") or key == 27:
        #alive = False