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
from LettersNumbers import images_to_strings, images_to_strings_easyocr
from LettersNumbers import extract_number_test
from def_sudoku_solver import solve_sudoku
from PaperDetMarkers import paper_markers
from render import render

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

#This function was used before I introducted parallel programming
def process_squares(squares, frame):
    result_array = [[None for _ in range(9)] for _ in range(9)]
    column_index = 0
    row_index = 0
        
    for i in squares:
        #plt.imshow(myImage[i[0][1]:i[1][1], i[0][0]:i[1][0]])
        #plt.show()
        #extract_number_test(myImage[i[0S][1]:i[1][1], i[0][0]:i[1][0]], "Blank", True)
        text = image_to_letter(frame[i[0][1]:i[1][1], i[0][0]:i[1][0]], True)
            
        result_array[row_index][column_index] = text
        row_index += 1
            
        if row_index == 9:  # Move to the next column after filling 8 rows
            row_index = 0
            column_index += 1
            
        # Stop if we fill the array
        if column_index == 9:
            break
    return result_array

def process_squares_parallel(frame, squares, character):
    
    images_2 = [None for _ in range(len(squares))]
        
    for idx, j in enumerate(squares):
        images_2[idx] = frame[j[0][1]:j[1][1], j[0][0]:j[1][0]]
    
    extracted_text = images_to_strings(images_2, character)
    print(extracted_text) 
    
    double_text = single_to_double_column_first(extracted_text)
    
    solved = solve_sudoku(double_text)
    print(f"Solved: {solved}")
    print(double_text)


if __name__ == '__main__':
    ###########Image Testing###########
    img = cv2.imread("WIN_20241202_17_10_34_Pro.jpg")
    myImage = img.copy()
    # in order to use, run preprocessing function
    roi, (x,y,w,h) = paper_markers(myImage)
    # send output of preprocessing function to getSquares to get list of rectangles
    squares = getSquares(roi, x, y, w, h)
    
    images_2 = [None for _ in range(len(squares))]
        
    for idx, j in enumerate(squares):
        images_2[idx] = myImage[j[0][1]:j[1][1], j[0][0]:j[1][0]]
    
    extracted_text = images_to_strings(images_2, True)
    print(extracted_text) 
    
    double_text = single_to_double_column_first(extracted_text)
    
    solved = solve_sudoku(double_text)
    print(f"Solved: {solved}")
    print(double_text)
    render(img, np.array(double_text).T.flatten())
    exit(0)

    #########Video Testing#############
    
    cap = cv2.VideoCapture(0)
    win_name = "Live Text Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to fetch new frame. Exiting.")
            break

        try:
            print(f"Processing frame at timestamp: {cv2.getTickCount()}") 
            linesP = preprocessing(frame)
            if linesP is None:
                print("Warning: preprocessing returned None. Skipping frame.")
                continue

            squares = getSquares(linesP)
            if not squares:
                print("No squares detected. Skipping frame.")
                continue

            result_array = process_squares_parallel(frame, squares, True)
            solved = solve_sudoku(result_array)
            print(f"Solved Sudoku: {solved}")
            print("This is a test")
            print(result_array)
        except Exception as e:
            print(f"Error processing frame: {e}")

        # Always display the frame, even if processing fails
        cv2.imshow(win_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting.")
            break

    cap.release()
    cv2.destroyAllWindows()
    

       
