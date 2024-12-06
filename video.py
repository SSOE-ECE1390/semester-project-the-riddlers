import cv2
from time import sleep
import numpy as np
import threading
from PaperDetMarkersVideo import paper_markers
from solve import solve

if __name__=='__main__':
    #first we need to detect the paper
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        exit(-1)
    markers_detected, roi, (x,y,w,h) = paper_markers(frame)
    sleep(1s)
    double_text = []
    avgConfidence = []
    solved = []
    threadPool = []
    for i in range(0,30):
        frame = cap.read()

        double_text.append([None])
        avgConfidence.append([None])
        solved.append([None])
        threadPool.append(threading.Thread(target=solve, args=(frame,double_text[i],avgConfidence[i],solved[i])))
    
    for i in threadPool:
        i.join()
    print(avgConfidence)
    #upon detecting the paper
        # wait 1s
        # attempt to solve the paper 30 times
    #render the solved puzzle until the paper no longer exists
