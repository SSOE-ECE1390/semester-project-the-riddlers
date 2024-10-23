import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re
from imutils import contours


#PyTesseract Lib https://pypi.org/project/pytesseract/
#Tesseract https://tesseract-ocr.github.io/tessdoc/Installation.html
#https://gist.github.com/qgolsteyn/7da376ced650a2894c2432b131485f5d

#Introduction: SO far the code works well in identifing words like "computer" and "babies", however it 
#doesn't work as well with identifying singel letters and numbers. The todo's will hopefully improve results. 

#Progress: Currently, we can detect letter and numbers frame by frame with low accuarcy with the detect_text funtion, 
#but its a start. The functoin also returns the box coordinates of each detected letter/number
#Additionally, a function that can draw a box around a detected letter/number => good for debugging in future work

#Problems: Sadly, LOTS. The frame by frame capture is very inconsistent in that every frame will detect different 
#words everytime, and can't detect letters very far away in the image.


#TODO: Identify single boxes in word search. Currently we have the grid without any numbers. My inital thinking was to then identify each 
#TODO: individual cell, and use pytesseract to identify the number/letter.
#TODO: Make it work better lol. Suggestions, feedback, and maybe even converting to more suceesfull method(trying something new) is appreciated!


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#Function to validate if the detected text is a real word or number (make sure there is no &*()^ etc)
def is_valid_text(text):
    # Only allow alphanumeric strings (words or numbers)
    return re.fullmatch(r'[a-zA-Z0-9]+', text) is not None


#Function to draw bounding boxes over detected text
def draw_boxes(frame, boxes):
    h, w, _ = frame.shape  #Get image dimensions
    for b in boxes:
        b = b.split(' ')
        x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        #Convert Tesseract's coordinates to OpenCV's format (y-coordinates are inverted)
        y = h - y
        y2 = h - y2
        #Draw a rectangle around each detected character
        cv2.rectangle(frame, (x, y2), (x2, y), (0, 255, 0), 2)
        

#Function to process the frame and detect text and retrieve box coor-> I haven't explored other functions in pytesseract and explored 
#minimally in using other libraries so this could be another path to explore
def detect_text(frame):

    thresh = process_grid_image(frame)

    #Get bounding boxes for each detected character
    boxes = pytesseract.image_to_boxes(thresh)

    #Apply OCR to the frame and split text into individual words
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(thresh, config=custom_config)
    words = text.split()

    #Initialize an empty list to store detected words/numbers in the current frame
    current_frame_words = []

    #Filter and add valid words/numbers to the current frame list
    for word in words:
        word = word.strip()  #Remove any leading/trailing spaces
        if is_valid_text(word):
            current_frame_words.append(word)

    return boxes, current_frame_words  #Return the character bounding boxes and current frame words



def process_grid_image(frame):
    #Convert the frame to grayscale (improves OCR accuracy)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,57,5)

    #Filter out all numbers and noise to isolate only boxes
    cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

    #Fix vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))  # Increase height for longer lines
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, vertical_kernel, iterations=2)
    
    #Fix horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))  # Increase width for longer lines
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)

    #Combine the horizontal and vertical lines
    grid_lines = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 1, 0)
    
    grid_lines = 255 - grid_lines
    
    return grid_lines


#########Testing#########

img = cv2.imread(os.path.relpath('data/sud.jpg'))

thresh = process_grid_image(img)

plt.imshow(img)
plt.show()
plt.imshow(thresh, cmap='gray')
plt.show()



#Initialize the video capture (0 is the default webcam)

cap = cv2.VideoCapture(0)
win_name = "Live Text Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
while True: # This code is taken from Homework 4
    #Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break

    #Detect text and get bounding boxes and current frame words
    boxes, current_frame_words = detect_text(frame)

    #Draw bounding boxes on the frame
    draw_boxes(frame, boxes.splitlines())

    #Display the frame with bounding boxes
    cv2.imshow(win_name, frame)

    #Print the current words detected in the frame
    print("Current words:", current_frame_words)

    #Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the capture when done
cap.release()
cv2.destroyAllWindows()
"""