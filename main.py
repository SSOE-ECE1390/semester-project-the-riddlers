import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re
from pprint import pprint
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colormaps
from sklearn.cluster import KMeans
from getSquares import preprocessing
from getSquares import getSquares
from LettersNumbers import image_to_letter
from LettersNumbers import extract_number_test
from def_sudoku_solver import solve_sudoku


if __name__ == '__main__':
    img = cv2.imread("testImage.png")
    
    myImage = img.copy()
    # in order to use, run preprocessing function
    linesP = preprocessing(myImage)
    # send output of preprocessing function to getSquares to get list of rectangles
    squares = getSquares(linesP)
    
    for i in squares:
        #plt.imshow(myImage[i[0][1]:i[1][1], i[0][0]:i[1][0]])
        #plt.show()
        extract_number_test(myImage[i[0][1]:i[1][1], i[0][0]:i[1][0]], "Blank", True)
        #text = image_to_letter(myImage[i[0][1]:i[1][1], i[0][0]:i[1][0]], True)
        #print(text)


        
