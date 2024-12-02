import PaperDetMarkers
import getSquares
import matplotlib.pyplot as plt
import cv2

if __name__=='__main__':
    img = cv2.imread("WIN_20241202_17_10_34_Pro.jpg")
    myImage = img.copy()
    img, (x,y,w,h) = PaperDetMarkers.paper_markers(img)
    plt.imshow(img)
    plt.show()
    
    squares = getSquares.getSquares(img, x, y, w, h)

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
