import cv2
import numpy as np
import pyautogui
from PIL import ImageGrab, Image
import matplotlib.pyplot as plt
import time
import pytesseract as tess
tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


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

    return_img = img

    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        ratio = w / h
        
        if area > MIN_AREA \
            and w > MIN_WIDTH and h > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
            contours_list.append({
                'contour': contour,
                'x': x,
                'y': y,
                'w': w,
                'h': h,
                'cx': x + (w / 2),
                'cy': y + (h / 2),
            })
            cv2.rectangle(return_img, (x, y), (x + w, y + h), (255, 255, 255), 1)

    return return_img

while True:
    capture = cv2.imread('my_card.jpg')
    merged_captures = np.array(capture)
    merged_captures = make_contours(merged_captures)
    cv2.imshow("", merged_captures)
    
    
    if cv2.waitKey(1) == 27:
        break

cv2.destroyWindow("")


        



