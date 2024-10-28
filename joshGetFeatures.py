import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colormaps

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
    #m2x+b2=m1x+b1
    #b2-b1=m1x-m2x
    #(b2-b1)/(m1-m2)=x
    #y+line1[1] = slope1*x+intercept1
    #line1[2]=m(line1[1])+b
    slope1 = (line1[1]-line1[3])/(line1[0]-line1[2])
    intercept1 = line1[1]-slope1*line1[0]#-line1[1]
    print(f"line1[0]: {line1[0]}")
    print(f"line1[1]: {line1[1]}")
    print(f"line1[2]: {line1[2]}")
    print(f"line1[3]: {line1[3]}")
    print(f"slope1: {slope1}")
    print(f"intercept1: {intercept1}")

    slope2 = (line2[1]-line2[3])/(line2[0]-line2[2])
    intercept2 = line2[1]-slope2*line2[0]#-line2[1]
    print(f"line2[0]: {line2[0]}")
    print(f"line2[1]: {line2[1]}")
    print(f"line2[2]: {line2[2]}")
    print(f"line2[3]: {line2[3]}")
    print(f"slope2: {slope2}")
    print(f"intercept2: {intercept2}")
    #test = img.copy()
    #test = cv2.line(test, (line1[0], line1[1]), (line1[2], line1[3]), (0,0,255), 3, cv2.LINE_AA)
    #test = cv2.line(test, (line2[0], line2[1]), (line2[2], line2[3]), (0,0,255), 3, cv2.LINE_AA)

    #horizontal line cases
    if(line1[1]-line1[3]==0):
        test = cv2.circle(test, (line1[1].astype(np.int32),(slope2*line1[1]+intercept2).astype(np.int32)), 30, (255,0,0), 30, cv2.LINE_AA)
        plt.imshow(test)
        plt.show()
        return (line1[1].astype(np.int32),(slope2*line1[1]+intercept2).astype(np.int32))
    if(line2[1]-line2[3]==0):
        test = cv2.circle(test, (line2[1].astype(np.int32),(slope1*line2[1]+intercept1).astype(np.int32)), 30, (255,0,0), 30, cv2.LINE_AA)
        plt.imshow(test)
        plt.show()
        return (line2[1].astype(np.int32),(slope1*line2[1]+intercept1).astype(np.int32))
    x = (intercept2-intercept1) / (slope1-slope2)
    y = slope1*x+intercept1
    print(x)
    print(y)
    test = cv2.circle(test, (x.astype(np.int32),y.astype(np.int32)), 30, (255,0,0), 30, cv2.LINE_AA)
    print(x.astype(np.int32))
    print(y.astype(np.int32))
    test = cv2.circle(test, (250,250), 30, (255,0,0), 30, cv2.LINE_AA)
    plt.imshow(test)
    plt.show()
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
    verticalLines.sort(key=lambda x:x[1])
    horizontalLines.sort(key=lambda x:x[0])
    print(verticalLines)
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
    squares = []
    test = img.copy()
    for i in range(0,len(verticalLines)-1):
        for j in range(0,len(horizontalLines)-1):
            point1 = calcIntercept(verticalLines[i], horizontalLines[j])
            #print(point1)
            point4 = calcIntercept(verticalLines[i+1], horizontalLines[j+1])
            #print(point4)
            if((point1[0]-point4[0]>100) and (point1[1]-point4[1]>100)):
                test = cv2.rectangle(test, point1, point4, (255,0,0), 3)
            squares.append((point1, point4))
    plt.imshow(test)
    plt.show()
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
    #print(i[0])
    #print(i[1])
    myImage = cv2.rectangle(myImage, i[0], i[1], (255,0,0), 3)

plt.imshow(myImage)
plt.show()

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
