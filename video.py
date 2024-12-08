import cv2
from time import sleep
import numpy as np
import threading
from PaperDetMarkersVideo import paper_markers
from solve import solve, solveSimple, assumedSolved
import matplotlib.pyplot as plt
from render import render

def exitProgram(cap, code=0):
    cap.release()
    cv2.destroyAllWindows()
    exit(code)

def renderLoop(frame, inputFrame):
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while True:
        ret, frame[0] = cap.read()
        if not ret:
            exitProgram(cap, -1)
        if type(inputFrame[0])==type(None):
            cv2.imshow('frame', frame[0])
        else:
            cv2.imshow('frame', inputFrame[0])
        if cv2.waitKey(1) == ord('q'):
            exitProgram(cap)

def doThings(frame, outputFrame):
    while True:
        while type(frame[0])==type(None):
            None
        double_text = []
        avgConfidence = []
        solved = []
        success = False
        while(success!=True):
            curFrame = frame[0]
            out = solveSimple(curFrame)
            if not type(out) == type(-1):
                success = True
            else:
                continue
            double_text.append(out[0])
            avgConfidence.append(out[1])
            solved.append(out[2])
        img = cv2.imread("WIN_20241202_17_10_34_Pro.jpg")
        while True:
            out = render(frame[0], np.array(double_text[0]).T.flatten())
            if type(out)==type(-1) and out==-1:
                print("We are failing")
                break
            outputFrame[0] = out


if __name__=='__main__':
    frame = [None]
    inputFrame = [None]
    renderThread = threading.Thread(target=renderLoop, args=(frame,inputFrame))
    renderThread.start()
    #renderLoop(frame)
    doThings(frame, inputFrame)
    exit(0)
    #first we need to detect the paper
    # this works
    #cap = cv2.VideoCapture(0)
    #markers_detected = False
    #while not markers_detected:
    #    ret, frame = cap.read()
    #    print("here")
    #    if not ret:
    #        exitProgram(cap, -1)
    #    cv2.imshow('frame', frame)
    #    markers_detected, roi, (x,y,w,h) = paper_markers(frame)
    #    if cv2.waitKey(1) == ord('q'):
    #        exitProgram(cap)
    #sleep(1)
    double_text = []
    avgConfidence = []
    solved = []
    threadPool = []
    for i in range(0,30):
        #ret, frame = cap.read()
        #cv2.imshow('frame', frame)
        #if cv2.waitKey(1) == ord('q'):
        #    exitProgram(cap)
        frame = cv2.imread("WIN_20241202_17_10_34_Pro.jpg")

        solve(frame, double_text[i],avgConfidence[i],solved[i])
        #double_text.append([None])
        #avgConfidence.append([None])
        #solved.append([None])
        #threadPool.append(threading.Thread(target=solve, args=(frame,double_text[i],avgConfidence[i],solved[i])))
        #threadPool[i].start()
    
    print(avgConfidence)
    print(double_text)
    print(solved)
    #cap.release()
    cv2.destroyAllWindows()
    #upon detecting the paper
        # wait 1s
        # attempt to solve the paper 30 times
    #render the solved puzzle until the paper no longer exists
