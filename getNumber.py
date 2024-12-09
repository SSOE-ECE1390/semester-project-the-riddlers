import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
import re
from imutils import contours
from multiprocessing import Pool
import easyocr

INTENSITY_THRESHOLD_HIGH = 250
INTENSITY_THRESHOLD_LOW = 5


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocessing(img):
    B,G,R = cv2.split(img)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    anded = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    anded[(R < 50) & (B < 50)] = 255
    anded = cv2.medianBlur(anded,5)
    ret3,anded = cv2.threshold(anded,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    anded = cv2.morphologyEx(anded,cv2.MORPH_OPEN,np.ones((5,5)),iterations=1)

    #anded = cv2.bitwise_and(R, cv2.bitwise_not(anded))
    #anded = cv2.medianBlur(anded,5)
    #ret3,anded = cv2.threshold(anded,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #anded = cv2.morphologyEx(anded,cv2.MORPH_OPEN,np.ones((5,5)),iterations=1)
    #plt.imshow(anded)
    #plt.show()
    return anded

def process_image(args):
    img = args
    return image_to_letter(img)


def process_text_with_confidence(image, config):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
    texts = data['text']
    confidences = data['conf']
    results = []

    for i, text in enumerate(texts):
        if text.strip():  # Ignore empty results
            results.append((text.strip(), int(confidences[i])))

    return results
    
def image_to_letter(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edgeCut = 10

    img = img[edgeCut:-edgeCut, edgeCut:-edgeCut]
    intensity = np.mean(img)
    if intensity > INTENSITY_THRESHOLD_HIGH and intensity < INTENSITY_THRESHOLD_LOW:
        return -1
       
    all_results = []
    config = '--psm 10 -c tessedit_char_whitelist=0123456789'
    results = process_text_with_confidence(img, config)
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
 
def images_to_strings(images):
    
    # Create arguments for each image
    args_list = [(img) for img in images]
    
    # Use multiprocessing Pool to process images in parallel
    with Pool() as pool:
        results = pool.map(process_image, args_list)
    
    return results
 


if __name__=='__main__':
    myImage = cv2.imread("WIN_20241209_07_42_32_Pro.jpg")
    plt.imshow(myImage)
    plt.show()
    out = preprocessing(myImage)
    plt.imshow(out)
    plt.show()
