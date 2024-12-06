from getSquares import getSquares
from PaperDetMarkers import paper_markers
from LettersNumbers import images_to_strings
from sudoku_solver import solve_sudoku
import cv2
import numpy as np

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


def solve(myImage, double_text, avgConfidence, solved):
    myImage = myImage.copy()
    roi, (x,y,w,h) = paper_markers(myImage)
    # send output of preprocessing function to getSquares to get list of rectangles
    squares = getSquares(roi, x, y, w, h)
    
    images_2 = [None for _ in range(len(squares))]
        
    for idx, j in enumerate(squares):
        images_2[idx] = myImage[j[0][1]:j[1][1], j[0][0]:j[1][0]]
    
    images_to_strings_out = images_to_strings(images_2, True)
    print(images_to_strings_out) 
    extracted_text = []
    confidences = []
    for i in extracted_text:
        if type(i)==tuple:
            extracted_text.append(i[0])
            confidences.append(i[1])
        else:
            extracted_text.append(i)
            confidences.append(-1)

    
    double_text[0] = single_to_double_column_first(extracted_text)
    
    solved[0] = solve_sudoku(double_text[0])
    avgConfidence[0] = np.mean(confidences)
    return double_text[0], avgConfidence[0], solved[0]

if __name__=='__main__':
    img = cv2.imread("WIN_20241202_17_10_34_Pro.jpg")
    solve(img, [None], [None], [None])

