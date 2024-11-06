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

img = cv2.imread("testImage.png")
B,G,R = cv2.split(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#get anything white, and make it even more white, take anything not white and make it black

th,thresh1_img = cv2.threshold(B,170,255,cv2.THRESH_BINARY)
th,thresh2_img = cv2.threshold(G,170,255,cv2.THRESH_BINARY)
th,thresh3_img = cv2.threshold(R,170,255,cv2.THRESH_BINARY)

anded = cv2.bitwise_and(thresh1_img, thresh2_img, thresh3_img)

plt.imshow(anded, cmap='gray')
plt.show()

plt.imshow(abs(255-anded), cmap='gray')
plt.show()

# use contours to get the rest

contours,hierarchy = cv2.findContours(anded,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

#plt.imshow(anded, cmap='gray')
#plt.show()

sorted_contour = sorted(contours,key=cv2.contourArea)  # smallest to largest

mask = np.zeros([img.shape[0],img.shape[1]],dtype='uint8')

mask=cv2.drawContours(mask,sorted_contour,-1,1,-1) # The 3rd entry is the index of the contour to use, -1 means all
                                                   # the 4th entry is the color to use. SInce this is a mask, just use 1
                                                   # the 5th entry is the linewidth. -1 will fill in the contour 

im_masked = cv2.bitwise_and(img,img,mask=mask)

#plt.imshow(im_masked)
#plt.show()

color = ['#641E16','b','g','r','c','m','y','#E67E22']

print(img.shape)
lines = np.zeros([img.shape[0], img.shape[1], img.shape[2]])

# Lets draw the lines around each coin
for idx in range(0,len(sorted_contour)):
    linewidth=5
    c=tuple(255*np.array(colormaps.to_rgb(color[3]))) 
    lines=cv2.drawContours(lines,sorted_contour,idx,(255,255,255),linewidth)
    im_masked=cv2.drawContours(im_masked,sorted_contour,idx,c,linewidth)

area=np.zeros((len(sorted_contour)))
for i in range(0,len(sorted_contour)):
    area[i]=cv2.contourArea(sorted_contour[i])

#plt.imshow(im_masked)
#plt.show()

# next steps are to compute a distance based on corner features

#plt.imshow(lines)
#plt.show()

result = img.copy()

lines = cv2.cvtColor(lines.astype(np.uint8), cv2.COLOR_RGB2GRAY)

plt.imshow(lines)
plt.show()

outline = cv2.Canny(lines, 80, 150)

linesP = cv2.HoughLinesP(lines, 1, np.pi / 180, 50, None, 400, 20)

def checkOrientation(line):
    vertical = abs(line[0]-line[2])
    horizontal = abs(line[1]-line[3])
    if(vertical>horizontal):
        print("Is vertical")
        #line is vertical
    if(horizontal>vertical):
        print("Is horizontal")
        #line is horizontal
    print(vertical)
    print(horizontal)

def calcIntercept(line1, line2):
    slope1 = (line1[1]-line1[3])/(line1[0]-line1[2])
    intercept1 = line1[1]-slope1*line1[0]#-line1[1]

    slope2 = (line2[1]-line2[3])/(line2[0]-line2[2])
    intercept2 = line2[1]-slope2*line2[0]#-line2[1]

    print()
    print(slope1)
    print(intercept1)
    print(slope2)
    print(intercept2)
    print()

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
        #if(vertical==0 or horizontal==0):
        #    pass
        if(vertical>horizontal):
            verticalLines.append(line)
        if(vertical<horizontal):
            horizontalLines.append(line)
    # remove outliers
    verticalLines = reject_outliers(verticalLines, 0, 1.5)
    horizontalLines = reject_outliers(horizontalLines, 1, 1.5)
    test = img.copy()
    for i in verticalLines:
        test = cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA)
    plt.imshow(test)
    plt.show()
    print(horizontalLines)
    test = img.copy()
    for i in horizontalLines:
        test = cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA)
    plt.imshow(test)
    plt.show()

    # end remove outliers
    # kmeans to find points
    verticalPoints = np.array(verticalLines)[:,0:2]
    horizontalPoints = np.array(horizontalLines)[:,0:2]
    verticalPoints[:,1] = 0
    horizontalPoints[:,0] = 0
    print(verticalLines)
    print(horizontalLines)
    print(verticalPoints)
    print(horizontalPoints)
        #preprocessing is finished
    verticalFit = KMeans(10).fit(verticalPoints)
    horizontalFit = KMeans(10).fit(horizontalPoints)
    
    verticalPredictions = verticalFit.predict(verticalPoints)
    horizontalPredictions = horizontalFit.predict(horizontalPoints)
    verticalGroups = [[],[],[],[],[],[],[],[],[],[]]
    horizontalGroups = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(0, len(verticalLines)):
        verticalGroups[verticalPredictions[i]].append(verticalLines[i])
    for i in range(0, len(horizontalLines)):
        horizontalGroups[horizontalPredictions[i]].append(horizontalLines[i])
    test = img.copy()
    for i in horizontalGroups[0]:
        test = cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA)
    plt.imshow(test)
    plt.show()
    # end kmeans

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

    test = img.copy()
    for i in verticalMeanLines:
        test = cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA)
    plt.imshow(test)
    plt.show()

    test = img.copy()
    for i in horizontalMeanLines:
        test = cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA)
    plt.imshow(test)
    plt.show()

    #end averaging

    verticalLines = verticalMeanLines
    horizontalLines = horizontalMeanLines

    verticalLines.sort(key=lambda x:x[0])
    horizontalLines.sort(key=lambda x:x[1])
    #test = img.copy()
    #for i in verticalLines:
    #    plt.imshow(cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA))
    #    plt.show()
    #test = img.copy()
    #for i in horizontalLines:
    #    plt.imshow(cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA))
    #    plt.show()

    #    print(i)
    #    if(verticalLines[i][1]-verticalLines[i+1][1]<1):
    #        print(verticalLines[i])
    #        del verticalLines[i]
    #        i-=1;
    #for i in range(0,len(horizontalLines)-1):
    #    print(i)
    #    if(horizontalLines[i][0]-horizontalLines[i+1][0]<1):
    #        del horizontalLines[i]
    #        i-=1;
    #print(verticalLines)
    #test = img.copy()
    #for i in verticalLines:
    #    test = cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA)
    #plt.imshow(test)
    #plt.show()
    #print(horizontalLines)
    #test = img.copy()
    #for i in horizontalLines:
    #    test = cv2.line(test, (i[0], i[1]), (i[2], i[3]), (0,0,255), 3, cv2.LINE_AA)
    #plt.imshow(test)
    #plt.show()
    #print("here")
    squares = []
    test = img.copy()
    print(len(verticalLines))
    print(len(horizontalLines))
    for i in range(0,len(verticalLines)-1):
        for j in range(0,len(horizontalLines)-1):
            print(j)
            point1 = calcIntercept(verticalLines[i], horizontalLines[j])
            #print(point1)
            point4 = calcIntercept(verticalLines[i+1], horizontalLines[j+1])
            #print(point4)
            #if((point1[0]-point4[0]>100) and (point1[1]-point4[1]>100)):
            #    test = cv2.rectangle(test, point1, point4, (255,0,0), 3)
            #test = cv2.circle(test, point1, 30, (255,0,0), 30, cv2.LINE_AA)
            #plt.imshow(test)
            #plt.show()
            #test = cv2.circle(test, point4, 30, (255,0,0), 30, cv2.LINE_AA)
            #plt.imshow(test)
            #plt.show()
    #test = cv2.circle(test, (x.astype(np.int32),y.astype(np.int32)), 30, (255,0,0), 30, cv2.LINE_AA)
            squares.append((point1, point4))
    #plt.imshow(test)
    #plt.show()
    return squares


if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(result, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

#def calcIntersections(linesP):
#    #linesNotCalculated
#    #for each line in linesNotCalculated
#        #Check every other line
#            #check if intercetp
#                #if intercept get intercept
#    for i in range(0, len(linesNotCalculated)):
#        for i in range(i, len(linesNotCalculated)):
#            
#    if linesP is not None:
#        for i in range(0, len(linesP)):
#            l = linesP[i][0]



myImage = img.copy()

squares = getSquares(linesP)
for i in squares:
    print(i)
    #print(i[0])
    #print(i[1])
    myImage = cv2.rectangle(myImage, i[0], i[1], (255,0,0), 3)
    #plt.imshow(myImage)
    #plt.show()
    #for j in i:
    #    if(not (j[0]<0 or j[1]<0 or j[0]>2160 or j[1]>3840)):
    #        print("here")
    #        myImage = cv2.rectangle(myImage, i[0], i[1], (255,0,0), 3)

plt.imshow(myImage)
plt.show()
cv2.imwrite("squares.png", myImage)

plt.imshow(outline,cmap='gray')
plt.show()

plt.imshow(result)
plt.show()

#plt.imshow(outlineHorz,cmap='gray')
#plt.show()
#plt.imshow(outlineVert,cmap='gray')
#plt.show()
#plt.imshow(lines)
#plt.show()
#
#plt.imshow(img_gray, cmap='gray')
#plt.show()
