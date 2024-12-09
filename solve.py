from getSquares import getSquares
from PaperDetMarkersVideo import paper_markers
#from LettersNumbers import images_to_strings, images_to_strings_debug
from sudoku_solver import solve_sudoku
import cv2
import numpy as np
import matplotlib.pyplot as plt
import getNumber
from getNumber import images_to_strings

cache = {}

#Converts a singe array of anysize above 81 to a 9x9 array
def single_to_double_column_first(single_array):
    if len(single_array) < 81:
        raise ValueError("Input array must have at least 81 elements.")
    
    double_array = [[None for _ in range(9)] for _ in range(9)]
    
    for idx in range(81):  
        row = idx % 9      
        col = idx // 9     
        double_array[row][col] = single_array[idx]
    
    return double_array

def solveSimple(myImage):
    myImage = myImage.copy()
    markers_detected, roi, (x,y,w,h), ids = paper_markers(myImage)
    if not markers_detected:
        return -2
    print("markers detected")
    key = (ids[0][0], ids[1][0], ids[2][0], ids[3][0])
    if key in cache:
        return cache[key]
    print(ids[0])
    # send output of preprocessing function to getSquares to get list of rectangles
    print("starting getSquares")
    squares = getSquares(roi, x, y, w, h)
    print("ending getSquares")
    if squares==-1:
        return -1
    trueSquares = []
    for i in squares:
        if((i[1][1] - i[0][1]) > 30 and (i[1][0]-i[0][0]) > 30):
            trueSquares.append(i)
    squares = trueSquares
    if(not len(squares)>=81):
        return -1
    images_2 = [None for _ in range(len(squares))]

    lettersNumbersImage = getNumber.preprocessing(myImage)

    for idx, j in enumerate(squares):
        images_2[idx] = lettersNumbersImage[j[0][1]:j[1][1], j[0][0]:j[1][0]]
    
    print("Starting tesseract")
    images_to_strings_out = images_to_strings(images_2)
    print("Finished tesseract")
    #print(images_to_strings_out) 
    extracted_text = []
    confidences = []
    for i in images_to_strings_out:
        print(i)
        if type(i)==tuple:
            extracted_text.append(int(i[0]))
            confidences.append(i[1])
        else:
            extracted_text.append(-1)

    
    #print(extracted_text)
    double_text = single_to_double_column_first(extracted_text)
    #print(double_text)
    
    print("Starting Sudoku")
    print(double_text)
    solved = solve_sudoku(double_text)
    print("Finished Sudoku")
    if not solved:
        return -3
    print(solved)
    print(double_text)
    #print(double_text)
    avgConfidence = np.mean(confidences)
    results = (double_text, avgConfidence, solved)
    cache[key]=results
    return double_text, avgConfidence, solved

def assumedSolved(myImage):
    double_text = [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]]
    avgConfidence = 0
    solved=True
    return double_text, avgConfidence, solved


def solve(myImage, double_text, avgConfidence, solved):
    print(double_text)
    print(avgConfidence)
    print(solved)
    myImage = myImage.copy()
    roi, (x,y,w,h) = paper_markers(myImage)
    # send output of preprocessing function to getSquares to get list of rectangles
    squares = getSquares(roi, x, y, w, h)
    trueSquares = []
    print("We have finished with the squares")


    B, G, R = myImage.split()   
    if squares==-1:
        return -1
    for i in squares:
        if(i[1][1] - i[0][1] > 30 and i[1][0]-i[0][0] > 30):
            trueSquares.append(i)
    print("The squares are working")

    JustRed = cv2.bitwise_xor(R,cv2.bitwise_or(G, B))
    
    images_2 = [None for _ in range(len(squares))]
        
    for idx, j in enumerate(squares):
        images_2[idx] = myImage[j[0][1]:j[1][1], j[0][0]:j[1][0]]
    
    print("Starting pytesseract")
    images_to_strings_out = images_to_strings(images_2, True)
    print("Finished pytesseract")
    #print(images_to_strings_out) 
    extracted_text = []
    confidences = []
    for i in images_to_strings_out:
        if type(i)==tuple:
            extracted_text.append(int(i[0]))
            confidences.append(i[1])
        else:
            extracted_text.append(-1)

    
    #print(extracted_text)
    double_text[0] = single_to_double_column_first(extracted_text)
    #print(double_text)
    
    print("Starting solve sudoku")
    print(double_text[0])
    solved[0] = solve_sudoku(double_text[0])
    print("Finished solve sudoku")
    #print(double_text)
    avgConfidence[0] = np.mean(confidences)
    return double_text[0], avgConfidence[0], solved[0]

if __name__=='__main__':
    img = cv2.imread("WIN_20241202_17_10_34_Pro.jpg")
    double_text = [[None]]
    print(solveSimple(img))
    print(double_text)


