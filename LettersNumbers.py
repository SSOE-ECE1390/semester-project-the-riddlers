import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re
from imutils import contours
from multiprocessing import Pool
import easyocr


#PyTesseract Lib https://pypi.org/project/pytesseract/
#Tesseract https://tesseract-ocr.github.io/tessdoc/Installation.html
#https://gist.github.com/qgolsteyn/7da376ced650a2894c2432b131485f5d
INTENSITY_THRESHOLD_HIGH = 250
INTENSITY_THRESHOLD_LOW = 5

############This file was used for testing diffrenet types of filters, easyocr, and pytesseract for this project. To see the final preprocessing, look into getNumber.py.################

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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
    gray = frame
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
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

#Function to convert image to string
def process_text(frame):
    custom_config = '--psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    text = pytesseract.image_to_string(frame, config=custom_config)
    return text

def process_text_with_confidence(image, config):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
    texts = data['text']
    confidences = data['conf']
    results = []

    for i, text in enumerate(texts):
        if text.strip():  # Ignore empty results
            results.append((text.strip(), int(confidences[i])))

    return results


#This function is used for testing
def extract_number_test(img, test, character):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    # Preprocessing
    denoised_image = cv2.GaussianBlur(gray, (3, 3), 0)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    sharpened_image = cv2.addWeighted(denoised_image, 1.5, blurred, -0.5, 0)
    adaptive = cv2.adaptiveThreshold(sharpened_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, binary_image = cv2.threshold(sharpened_image, 75, 255, cv2.THRESH_BINARY)

    # Display preprocessing results
    print(binary_image.shape)
    center_x = binary_image.shape[0]//2
    center_y = binary_image.shape[1]//2
    check_blank = binary_image[center_x-30:center_x+30, center_y-30:center_y+30]
    intensity = np.mean(check_blank)

    center_x = denoised_image.shape[0]//2
    center_y = denoised_image.shape[1]//2
    #denoised_image = denoised_image[center_x-30:center_x+30, center_y-30:center_y+30]
    
    center_x = blurred.shape[0]//2
    center_y = blurred.shape[1]//2
    Eblurred = blurred[center_x-30:center_x+30, center_y-30:center_y+30]
    
    center_x = sharpened_image.shape[0]//2
    center_y = sharpened_image.shape[1]//2
    #sharpened_image = sharpened_image[center_x-30:center_x+30, center_y-30:center_y+30]
    
    center_x = binary_image.shape[0]//2
    center_y = binary_image.shape[1]//2
    #binary_image = binary_image[center_x-30:center_x+30, center_y-30:center_y+30]
    
    
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 6, 1), plt.imshow(denoised_image, cmap='gray'), plt.title('Denoised')
    plt.subplot(1, 6, 2), plt.imshow(blurred, cmap='gray'), plt.title('Blurred')
    plt.subplot(1, 6, 3), plt.imshow(sharpened_image, cmap='gray'), plt.title('Sharpened')
    plt.subplot(1, 6, 4), plt.imshow(binary_image, cmap='gray'), plt.title('Binary')
    plt.subplot(1, 6, 5), plt.imshow(adaptive, cmap='gray'), plt.title('Adaptive')
    plt.subplot(1, 6, 6), plt.imshow(check_blank, cmap='gray'), plt.title('Intensity')
    plt.show()
    
    print(f"Intensity {intensity}")
    if intensity > INTENSITY_THRESHOLD_HIGH or intensity < INTENSITY_THRESHOLD_LOW:
        print("\nNo Letter")
        return -1
    
    
    print(f"\nStart Test {test}")
    all_results = []
    if character:
        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
    else:
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for step_name, image in [("Denoised", denoised_image), 
                             ("Blurred", blurred), 
                             ("Sharpened", sharpened_image), 
                             ("Binary", binary_image)]:
        results = process_text_with_confidence(image, config)
        if results is None:
            print("No results returned by OCR.")
            continue
        print(f"{step_name} Image Results:")
        for text, confidence in results:
            print(f"  Text: {text}, Confidence: {confidence}")
            # Filter for single digits and confidence > 50
            if len(text) == 1 and confidence > 35:
                all_results.append((text, confidence))

    # Find the result with the highest confidence
    #if all_results:
    if len(all_results) != 0:
        best_result = max(all_results, key=lambda x: x[1])  # Sort by confidence
        print(f"\nBest Single Digit: {best_result[0]} with Confidence: {best_result[1]}")
        return best_result[0]  # Return the digit with the highest confidence
    else:
        print("\nCould Not Detect Character")
        return -1  # Return -1 if no valid digit is found
    
def image_to_letter(img, character):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img

    # Preprocessing
    denoised_image = cv2.GaussianBlur(gray, (3, 3), 0)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    sharpened_image = cv2.addWeighted(denoised_image, 1.5, blurred, -0.5, 0)
    #adaptive = cv2.adaptiveThreshold(sharpened_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, binary_image = cv2.threshold(sharpened_image, 75, 255, cv2.THRESH_BINARY)
    
    center_x = binary_image.shape[0]//2
    center_y = binary_image.shape[1]//2
    check_blank = binary_image[center_x-30:center_x+30, center_y-30:center_y+30]
    intensity = np.mean(check_blank)
    if intensity > INTENSITY_THRESHOLD_HIGH and intensity < INTENSITY_THRESHOLD_LOW:
        return -1
    
    center_x = binary_image.shape[0]//2
    center_y = binary_image.shape[1]//2
    #check_blank = binary_image[center_x-30:center_x+30, center_y-30:center_y+30]
    intensity = np.mean(check_blank)

    center_x = denoised_image.shape[0]//2
    center_y = denoised_image.shape[1]//2
    #denoised_image = denoised_image[center_x-30:center_x+30, center_y-30:center_y+30]
    
    center_x = blurred.shape[0]//2
    center_y = blurred.shape[1]//2
    #blurred = blurred[center_x-30:center_x+30, center_y-30:center_y+30]
    
    center_x = sharpened_image.shape[0]//2
    center_y = sharpened_image.shape[1]//2
    #sharpened_image = sharpened_image[center_x-30:center_x+30, center_y-30:center_y+30]
    
    center_x = binary_image.shape[0]//2
    center_y = binary_image.shape[1]//2
    #binary_image = binary_image[center_x-30:center_x+30, center_y-30:center_y+30]
    
    
    # Select the most effective preprocessed image
    # Based on experiments, sharpened or binary image is usually sufficient
    processed_images = [denoised_image, binary_image, blurred]
    
    all_results = []
    if character:
        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
    else:
        config = '--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for image in processed_images:
        results = process_text_with_confidence(image, config)
        for text, confidence in results:
            # Filter for single digits and confidence > 50
            if len(text) == 1 and confidence > 50:
                all_results.append((text, confidence))

    # Find the result with the highest confidence
    #if all_results:
    if len(all_results) != 0:
        best_result = max(all_results, key=lambda x: x[1])  # Sort by confidence
        print(f"\nBest Single Digit: {best_result[0]} with Confidence: {best_result[1]}")
        return best_result  # Return the digit with the highest confidence
    else:
        return -1  # Return -1 if no valid digit is found
    
def image_to_letter_easyocr(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    reader = easyocr.Reader(["en"])
    
    # Preprocessing
    denoised_image = cv2.GaussianBlur(gray, (3, 3), 0)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    sharpened_image = cv2.addWeighted(denoised_image, 1.5, blurred, -0.5, 0)
    _, binary_image = cv2.threshold(sharpened_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    center_x = binary_image.shape[0]//2
    center_y = binary_image.shape[1]//2
    check_blank = binary_image[center_x-30:center_x+30, center_y-30:center_y+30]
    intensity = np.mean(check_blank)
    if intensity == 255:
        return -1
    
    # Select the most effective preprocessed image
    # Based on experiments, sharpened or binary image is usually sufficient
    processed_images = [blurred, denoised_image]
    
    all_results = []
    for image in processed_images:
        results = reader.readtext(image)
        for (bbox, text, confidence) in results:
            # Filter for single digits and confidence > 50
            if len(text) == 1 and confidence > 50:
                all_results.append((text, confidence))

    # Find the result with the highest confidence
    #if all_results:
    if len(all_results) != 0:
        best_result = max(all_results, key=lambda x: x[1])  # Sort by confidence
        return best_result[0]  # Return the digit with the highest confidence
    else:
        return -1  # Return -1 if no valid digit is found
    
def process_image(args):
    img, char = args
    return image_to_letter(img, char)

def process_image_debug(args):
    img, char = args
    return extract_number_test(img,"", char)

def process_image_easyocr(args):
    img, char = args
    return image_to_letter_easyocr(img)

def images_to_strings_easyocr(images, character):
    
    # Create arguments for each image
    args_list = [(img, character) for img in images]
    
    # Use multiprocessing Pool to process images in parallel
    with Pool() as pool:
        results = pool.map(process_image_easyocr, args_list)
    
    return results

def images_to_strings_debug(images, character):
    
    # Create arguments for each image
    args_list = [(img, character) for img in images]
    
    # Use multiprocessing Pool to process images in parallel
    with Pool() as pool:
        results = pool.map(process_image_debug, args_list)
    
    return results
    
def images_to_strings(images, character):
    
    # Create arguments for each image
    args_list = [(img, character) for img in images]
    
    # Use multiprocessing Pool to process images in parallel
    with Pool() as pool:
        results = pool.map(process_image, args_list)
    
    return results
    

def change_res(cap,width,height):
    cap.set(3,width)
    cap.set(4,height)
    
def input_text(image, text):
    position = (image.shape[1]//2, image.shape[0]//2)  # (x, y) coordinates of the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 0, 0)  # BGR color (white in this case)
    thickness = 3
    
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    
    return image
    
if __name__ == '__main__':
    #########Testing#########
    img = cv2.imread(os.path.relpath('data/blanksquare.jpg'))
    image = input_text(img, 'A')
    plt.imshow(image)
    plt.show()

    img = cv2.imread(os.path.relpath('data/video_k.jpg'))
    process_text_with_confidence(img, "k")

    """
    img = cv2.imread(os.path.relpath('data/video_k.jpg'))
    extract_number_test(img, "k")
    img = cv2.imread(os.path.relpath('data/video_5.jpg'))
    extract_number_test(img, 5)
    img = cv2.imread(os.path.relpath('data/video_1.jpg'))
    extract_number_test(img, "1")
    img = cv2.imread(os.path.relpath('data/video_m.jpg'))
    extract_number_test(img, "m")
    img = cv2.imread(os.path.relpath('data/video_k_1.jpg'))
    extract_number_test(img, "k1")
    img = cv2.imread(os.path.relpath('data/video_L.jpg'))
    extract_number_test(img, "L")
    img = cv2.imread(os.path.relpath('data/video_a.jpg'))
    extract_number_test(img, "a")
    img = cv2.imread(os.path.relpath('data/video_2.jpg'))
    extract_number_test(img, "2")
    img = cv2.imread(os.path.relpath('data/video_3.jpg'))
    extract_number_test(img, "3")
    img = cv2.imread(os.path.relpath('data/video_7.jpg'))
    extract_number_test(img, "7")
    img = cv2.imread(os.path.relpath('data/video_9.jpg'))
    extract_number_test(img, "9")
    """

        

    #Initialize the video capture (0 is the default webcam)
    """"
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
