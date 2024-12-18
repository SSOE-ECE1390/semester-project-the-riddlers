import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colormaps
from sklearn.cluster import KMeans

#reject squares that don't meet the size requirements
def reject_outliers(data, ind, m=2.0):
    myData = np.array(data)
    relevant = np.array(data)[:,ind]
    return myData[(abs(relevant-np.mean(relevant)) < m*np.std(relevant))].tolist()

#processing for image detected
def newPreprocessing(img):
    B,G,R = cv2.split(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #plt.imshow(img_HSV)
    #plt.show()
    H,S,V = cv2.split(img_HSV)
    th,threshH1_img = cv2.threshold(H,110,255,cv2.THRESH_BINARY)
    #th,threshH2_img = cv2.threshold(H,130,255,cv2.THRESH_BINARY_INV)
    #plt.imshow(threshH2_img, cmap='gray')
    #plt.show()
    #hAnded = cv2.bitwise_and(threshH1_img, threshH2_img)
    #plt.imshow(hAnded, cmap='gray')
    #plt.show()
    #get anything white, and make it even more white, take anything not white and make it black
    
    th,thresh1_img = cv2.threshold(B,130,255,cv2.THRESH_BINARY)
    th,thresh2_img = cv2.threshold(G,130,255,cv2.THRESH_BINARY)
    th,thresh3_img = cv2.threshold(R,130,255,cv2.THRESH_BINARY)
    
    #anded2 = cv2.bitwise_and(threshH1_img, thresh1_img, thresh2_img, thresh3_img)
    
    #print("firstTest")
    #plt.imshow(thresh3_img)
    #plt.show()
    #anded = cv2.bitwise_and(thresh1_img, thresh2_img, thresh3_img)
    #print("secondTest")
    #plt.imshow(thresh3_img)
    #plt.show()
    anded = cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_and(thresh1_img, thresh2_img), thresh3_img), threshH1_img)
    #print("here")
    #plt.imshow(originalAnded)
    #plt.show()
    #plt.imshow(anded2, cmap='gray')
    #plt.show()

    return anded

#Use an adaptive threshold and contours to detect the grid
def preprocessing(img):
    B,G,R = cv2.split(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #H = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[0]
    #plt.imshow(R)
    #plt.show()
    #print((R > 100) & (B < 100))
    #R[(R > 100) & (B < 100)] = 255
    #plt.imshow(R)
    #plt.show()
    #get anything white, and make it even more white, take anything not white and make it black
    
    #th,thresh1_img = cv2.threshold(B,170,255,cv2.THRESH_BINARY)
    #th,thresh2_img = cv2.threshold(G,170,255,cv2.THRESH_BINARY)
    #th,thresh3_img = cv2.threshold(R,170,255,cv2.THRESH_BINARY)
    #th,threshH_img = cv2.threshold(H,110,255,cv2.THRESH_BINARY)
    #plt.imshow(threshH_img)
    #plt.show()
    
    #anded = cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_and(thresh1_img, thresh2_img), thresh3_img), threshH_img)
    #anded = cv2.bitwise_and(cv2.bitwise_and(thresh2_img, thresh3_img), thresh1_img)
    anded = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    anded[(R > 100) & (B < 100)] = 255
    anded = cv2.medianBlur(anded,5)
    ret3,anded = cv2.threshold(anded,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    anded = cv2.morphologyEx(anded,cv2.MORPH_OPEN,np.ones((5,5)),iterations=1)
    #anded = cv2.bitwise_and(thresh1_img, threshH_img)
    #anded = cv2.bitwise_and(thresh1_img, thresh2_img, thresh3_img)
    
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


    #
    linesP = cv2.HoughLinesP(lines, 1, np.pi / 180, 50, None, 400, 20)
    print(linesP)
    
    return linesP

#Finds where the lines intercept to find the specific squares
def calcIntercept(line1, line2):
    slope1 = (line1[1]-line1[3])/(line1[0]-line1[2])
    intercept1 = line1[1]-slope1*line1[0]#-line1[1]

    slope2 = (line2[1]-line2[3])/(line2[0]-line2[2])
    intercept2 = line2[1]-slope2*line2[0]#-line2[1]

    x = (intercept2-intercept1) / (slope1-slope2)
    y = slope1*x+intercept1
    #vertical line cases
    if (abs(intercept1)>=2147483646):
        x = line1[0]
        y = slope2*x+intercept2
    if (abs(intercept2)>=2147483646):
        x = line2[0]
        y = slope1*x+intercept1
    #end vertical line cases

    return [x.astype(np.int32),y.astype(np.int32)]
    
# main function that takes the mean of all the detected hough lines and returns the corridnates for each of the squares
def getSquares(img, x, y, w, h):
    lines = preprocessing(img)
    #myTest = img.copy()
    #for i in lines:
    #    cv2.line(myTest, i[0][0:2], i[0][2:4], (255,0,0), 3)
    #plt.imshow(myTest)
    #plt.show()
    if type(lines) == type(None):
        return -1
    print("here")
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
    # get rid of lines that are close to the edge
    newVerticalLines = []
    newHorizontalLines = []
    lineLimitH = h/15
    lineLimitV = w/15
    for i in horizontalLines:
        if i[1]>lineLimitH and i[1]<h-lineLimitH and i[3]>lineLimitH and i[3]<h-lineLimitH:
            newHorizontalLines.append(i)
    for i in verticalLines:
        if i[0]>lineLimitV and i[0]<w-lineLimitV and i[2]>lineLimitV and i[2]<w-lineLimitV:
            newVerticalLines.append(i)

    horizontalLines = newHorizontalLines
    verticalLines = newVerticalLines




    # remove outliers
    #global test
    ##verticalLines = reject_outliers(verticalLines, 0, 1.5)
    ##horizontalLines = reject_outliers(horizontalLines, 1, 1.5)



    verticalLines.sort(key=lambda x:x[0])
    horizontalLines.sort(key=lambda x:x[1])
    #print(verticalLines)
    try:
        verticalPoints = np.array(verticalLines)[:,0]
        horizontalPoints = np.array(horizontalLines)[:,1]
    except:
        print(verticalLines)
        print(horizontalLines)
        return -1
    verticalGroups = [[]]
    verticalGroups[0].append(verticalLines[0])
    pos = 0
    for i in range(1, len(verticalPoints)):
        if(abs(verticalPoints[i] - verticalPoints[i-1])>30):
            verticalGroups.append([])
            pos+=1
        verticalGroups[pos].append(verticalLines[i])
    #print(verticalGroups)
    horizontalGroups = [[]]
    horizontalGroups[0].append(horizontalLines[0])
    pos = 0
    for i in range(1, len(horizontalPoints)):
        if(abs(horizontalPoints[i] - horizontalPoints[i-1])>30):
            horizontalGroups.append([])
            pos+=1
        horizontalGroups[pos].append(horizontalLines[i])
    #print(horizontalGroups)
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

    #lineTest = test.copy()
    #for i in verticalMeanLines:
    #    cv2.line(lineTest, i[0:2], i[2:4], (255,0,0), 3)

    #for i in horizontalMeanLines:
    #    cv2.line(lineTest, i[0:2], i[2:4], (255,0,0), 3)

    verticalLines.sort(key=lambda x:x[0])
    horizontalLines.sort(key=lambda x:x[1])
    squares = []
    for i in range(0,len(verticalLines)-1):
        for j in range(0,len(horizontalLines)-1):
            point1 = calcIntercept(verticalLines[i], horizontalLines[j])
            point4 = calcIntercept(verticalLines[i+1], horizontalLines[j+1])
            point1[0]+=x
            point1[1]+=y

            point4[0]+=x
            point4[1]+=y

            squares.append((point1, point4))
    return squares

if __name__=='__main__':
    img = cv2.imread("myTestImage1.png")
    print("here")
    
    myImage = img.copy()
    #test = newPreprocessing(myImage)
    #plt.imshow(test)
    #plt.show()
    #exit()
    # in order to use, run preprocessing function
    #linesP = preprocessing(myImage)
    #plt.imshow(linesP)
    #plt.show()
    # send output of preprocessing function to getSquares to get list of rectangles
    squares = getSquares(linesP)
    # all done

    test = img.copy()
    test = cv2.rectangle(test, squares[0][0], squares[0][1], (255,0,0), 3)
    plt.imshow(test)
    plt.show()
    #plt.imshow(myImage[squares[0][0][1]:squares[0][1][1], squares[0][0][0]:squares[0][1][0]])
    print(squares[0][0][0])#x
    print(squares[0][0][1])#y#bottom left
    print(squares[0][1][0])#x
    print(squares[0][1][1])#y#top right
    plt.show()
    for i in squares:
        #plt.imshow(myImage[i[0][1]:i[1][1], i[0][0]:i[1][0]])
        #plt.show()
        myImage = cv2.rectangle(myImage, i[0], i[1], (255,0,0), 3)
    plt.imshow(myImage)
    plt.show()
    

