from PaperDetMarkersVideo import paper_markers
from getSquares import getSquares
import cv2
import matplotlib.pyplot as plt
import os

    
def input_text(image, topleft, bottomright, text):
    top = topleft[0] 
    bottom = bottomright[0]
    left = topleft[1]
    right = bottomright[1]
    position = ((right+left)//2, (bottom+top)//2)
    #position = (image.shape[1]//2, image.shape[0]//2)  # (x, y) coordinates of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)  # BGR color (white in this case)
    thickness = 3
    
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    
    return image

def render(img, solvedPuzzle):
    plt.imshow(img)
    plt.show()
    markers_detected, roi, (x,y,w,h) = paper_markers(img)
    plt.imshow(roi)
    plt.show()
    squares = getSquares(roi, x,y,w,h)
    input_text(img, squares[0][0], squares[0][1], str(5))
    return True

if __name__=='__main__':
    img = cv2.imread(r"WIN_20241202_17_10_41_Pro.jpg")
    render(img, "this shit")
