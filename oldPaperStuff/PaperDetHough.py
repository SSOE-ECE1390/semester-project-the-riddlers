import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def threshold_paper(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(3,3),0)

    edges = cv2.Canny(image=blur,threshold1=100,threshold2=200) 

    imgC = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180,100,minLineLength=100,maxLineGap=50) #probabilistic Hough Transform
    #can adjust min line length, max line gap, need to consider how much paper covers screen and overlap with hands holding paper

    for line in lines:
            x1,y1,x2,y2 = line[0] #pull out all line indices

             #draw lines on image
            cv2.line(imgC, (x1,y1),(x2,y2), (0,0,255),2, cv2.LINE_AA)

    return imgC