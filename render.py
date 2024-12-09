from PaperDetMarkersVideo import paper_markers
from getSquares import getSquares
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

    
def input_text(image, topleft, bottomright, text):
    top = topleft[0] 
    bottom = bottomright[0]
    left = topleft[1]
    right = bottomright[1]
    print(top)
    print(bottom)
    print(left)
    print(right)
    position = ((bottom+top)//2, (right+left)//2)
    #position = (image.shape[1]//2, image.shape[0]//2)  # (x, y) coordinates of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)  # BGR color (white in this case)
    thickness = 3
    
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    
    return image

def render(img, solvedPuzzle):
    print(solvedPuzzle)
    print("Above is the solved Puzzle")
    myRender = img.copy()
    markers_detected, roi, (x,y,w,h) = paper_markers(img)
    if not markers_detected:
        return -2
    squares = getSquares(roi, x,y,w,h)
    if squares == -1:
        return -1
    for i in squares:
        myRender = cv2.rectangle(myRender, i[0], i[1], (255,0,0), 3)
    for i in range(0, 81 if len(squares) >= 81 else len(squares)):
        input_text(myRender,squares[i][0], squares[i][1], str(solvedPuzzle[i]))

    #for pos,i in enumerate(squares):
    #    input_text(img,i[0], i[1], str(solvedPuzzle[pos]))

    #for pos,i in enumerate(squares):
    #    input_text(img,i[0], i[1], str(solvedPuzzle[pos]))
    return myRender

if __name__=='__main__':
    img = cv2.imread(r"WIN_20241202_17_10_41_Pro.jpg")
    myRender = img.copy()
    solvedPuzzle=[[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]]
    solvedPuzzle = np.array(solvedPuzzle).T.flatten()
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    render(myRender, solvedPuzzle)
    out = render(myRender, solvedPuzzle)
    plt.imshow(out)
    plt.show()
    print("here")
    exit(0)
