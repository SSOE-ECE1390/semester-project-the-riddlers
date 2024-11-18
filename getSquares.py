import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colormaps
from sklearn.cluster import KMeans

def reject_outliers(data, ind, m=2.0):
    myData = np.array(data)
    relevant = np.array(data)[:,ind]
    return myData[(abs(relevant-np.mean(relevant)) < m*np.std(relevant))].tolist()

def preprocessing(img):
    B,G,R = cv2.split(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #get anything white, and make it even more white, take anything not white and make it black
    
    th,thresh1_img = cv2.threshold(B,170,255,cv2.THRESH_BINARY)
    th,thresh2_img = cv2.threshold(G,170,255,cv2.THRESH_BINARY)
    th,thresh3_img = cv2.threshold(R,170,255,cv2.THRESH_BINARY)
    
    anded = cv2.bitwise_and(thresh1_img, thresh2_img, thresh3_img)
    
    # use contours to get the rest
    
    contours,hierarchy = cv2.findContours(anded,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    
    sorted_contour = sorted(contours,key=cv2.contourArea)  # smallest to largest
    
    mask = np.zeros([img.shape[0],img.shape[1]],dtype='uint8')
    
    mask=cv2.drawContours(mask,sorted_contour,-1,1,-1) # The 3rd entry is the index of the contour to use, -1 means all
                                                       # the 4th entry is the color to use. SInce this is a mask, just use 1
                                                       # the 5th entry is the linewidth. -1 will fill in the contour 
    color = ['#641E16','b','g','r','c','m','y','#E67E22']
    
    lines = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    
    # Lets draw the lines around each coin
    for idx in range(0,len(sorted_contour)):
        linewidth=5
        c=tuple(255*np.array(colormaps.to_rgb(color[3]))) 
        lines=cv2.drawContours(lines,sorted_contour,idx,(255,255,255),linewidth)
    
    
    lines = cv2.cvtColor(lines.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    
    linesP = cv2.HoughLinesP(lines, 1, np.pi / 180, 50, None, 400, 20)

    return linesP

def calcIntercept(line1, line2):
    slope1 = (line1[1]-line1[3])/(line1[0]-line1[2])
    intercept1 = line1[1]-slope1*line1[0]#-line1[1]

    slope2 = (line2[1]-line2[3])/(line2[0]-line2[2])
    intercept2 = line2[1]-slope2*line2[0]#-line2[1]

    x = (intercept2-intercept1) / (slope1-slope2)
    print("hereaksdjflakj")
    y = slope1*x+intercept1
    #vertical line cases
    if (abs(intercept1)>=2147483646):
        x = line1[0]
        y = slope2*x+intercept2
    if (abs(intercept2)>=2147483646):
        x = line2[0]
        y = slope1*x+intercept1
    #end vertical line cases
    print(x.astype(np.int32))
    print(y.astype(np.int32))

    return (x.astype(np.int32),y.astype(np.int32))
    

def getSquares(lines):
    verticalLines = []
    horizontalLines = []
    for i in lines:
        line = i[0]
        horizontal = abs(line[0]-line[2])
        vertical = abs(line[1]-line[3])
        if(vertical>horizontal):
            verticalLines.append(line)
        if(vertical<horizontal):
            horizontalLines.append(line)
    # remove outliers
    verticalLines = reject_outliers(verticalLines, 0, 1.5)
    horizontalLines = reject_outliers(horizontalLines, 1, 1.5)

    verticalLines.sort(key=lambda x:x[0])
    horizontalLines.sort(key=lambda x:x[1])
    verticalPoints = np.array(verticalLines)[:,0]
    horizontalPoints = np.array(horizontalLines)[:,1]
    verticalGroups = [[]]
    pos = 0
    for i in range(1, len(verticalPoints)):
        if(abs(verticalPoints[i] - verticalPoints[i-1])>10):
            verticalGroups.append([])
            pos+=1
        verticalGroups[pos].append(verticalLines[i])
    horizontalGroups = [[]]
    pos = 0
    for i in range(1, len(horizontalPoints)):
        if(abs(horizontalPoints[i] - horizontalPoints[i-1])>10):
            horizontalGroups.append([])
            pos+=1
        horizontalGroups[pos].append(horizontalLines[i])
    # end seperate line bunches

        

    #start averaging
    verticalMeanLines = []
    horizontalMeanLines = []
    for i in verticalGroups:
        verticalMeanLines.append([
            np.mean(np.array(i)[:,0]).astype(np.int32),
            np.mean(np.array(i)[:,1]).astype(np.int32),
            np.mean(np.array(i)[:,2]).astype(np.int32),
            np.mean(np.array(i)[:,3]).astype(np.int32)
            ])
    for i in horizontalGroups:
        horizontalMeanLines.append([
            np.mean(np.array(i)[:,0]).astype(np.int32),
            np.mean(np.array(i)[:,1]).astype(np.int32),
            np.mean(np.array(i)[:,2]).astype(np.int32),
            np.mean(np.array(i)[:,3]).astype(np.int32)
            ])

    #end averaging

    verticalLines = verticalMeanLines
    horizontalLines = horizontalMeanLines

    verticalLines.sort(key=lambda x:x[0])
    horizontalLines.sort(key=lambda x:x[1])
    squares = []
    for i in range(0,len(verticalLines)-1):
        for j in range(0,len(horizontalLines)-1):
            print(j)
            point1 = calcIntercept(verticalLines[i], horizontalLines[j])
            point4 = calcIntercept(verticalLines[i+1], horizontalLines[j+1])
            squares.append((point1, point4))
    return squares

img = cv2.imread("testImage.png")

myImage = img.copy()
# in order to use, run preprocessing function
linesP = preprocessing(myImage)
# send output of preprocessing function to getSquares to get list of rectangles
squares = getSquares(linesP)
# all done
for i in squares:
    myImage = cv2.rectangle(myImage, i[0], i[1], (255,0,0), 3)

plt.imshow(myImage)
plt.show()
cv2.imwrite("squares.png", myImage)