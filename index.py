import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab, Image
import matplotlib.pyplot as plt
import time
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

is_reading_something = False
contours_list = []

def filter_img(img):
    img_blurred = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), ksize=(5, 5), sigmaX=0)
    img_blur_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )

    return img_blur_thresh

def make_contours(img):
    img_thresh = filter_img(img)
    height, width = img_thresh.shape
    
    contours, _ = cv2.findContours(
        img_thresh, 
        mode=cv2.RETR_TREE    , 
        method=cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, 1), dtype=np.uint8)
    
    global contours_list

    MIN_AREA = 1600
    MIN_WIDTH, MIN_HEIGHT = 700, 500

    return_img = img

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if w < MIN_WIDTH or h < MIN_HEIGHT or w * h < MIN_AREA:
            continue

        cv2.rectangle(return_img, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)
        
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
        text = f'({x + int(w / 2)}, {y + int(h / 2)})'

        tx, ty = cv2.getTextSize(text, font, 1, 2)[0]
        
        cv2.putText(return_img, text, (x + int(w / 2) - tx, y + int(h / 2) - ty), font, 2, 255)
        contours_list.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

    global is_reading_something
    if not contours_list:
        is_reading_something = False
    else:
        is_reading_something = True

    return return_img

while True:
    capture = ImageGrab.grab(bbox=(0, 0, 1920, 1080))
    merged_captures = cv2.cvtColor(np.array(capture), cv2.COLOR_BGR2RGB)
    merged_captures = make_contours(merged_captures)
    cv2.imshow("", merged_captures)
    
    print(is_reading_something)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyWindow("")


        



