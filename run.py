"""
    Name: Christian de Guzman
    Personal Project: American Sign Language Detector
"""

import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Constants
OFFSET = 20
IMG_SIZE = 300
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", 
          "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

def main():
    capture = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

    folder = "images/Y"
    counter = 0

    while True:
        success, img = capture.read()
        img_out = img.copy()
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox'] #bounding box for hand
            
            img_white = np.ones((IMG_SIZE, IMG_SIZE, 3), np.uint8)*255 
            
            img_crop = img[y-OFFSET:y + h+OFFSET, x-OFFSET:x + w+OFFSET]
            img_crop_shape = img_crop.shape
            
            aspect_ratio = h/w
            
            if aspect_ratio > 1:
                k = IMG_SIZE/h #constant
                width_calculated = math.ceil(k*w)
                image_resize = cv2.resize(img_crop, (width_calculated, IMG_SIZE))
                img_resize_shape = image_resize.shape
                width_gap = math.ceil((IMG_SIZE - width_calculated) / 2) #To help center rectangular shaped image
                img_white[:, width_gap:width_calculated+width_gap] = image_resize
                prediction, indx = classifier.getPrediction(img_white, draw=False)
                print(prediction, indx)
            else:
                k = IMG_SIZE/w #constant
                height_calculated = math.ceil(k*h)
                image_resize = cv2.resize(img_crop, (IMG_SIZE, height_calculated))
                img_resize_shape = image_resize.shape
                height_gap = math.ceil((IMG_SIZE - height_calculated) / 2) #To help center rectangular shaped image
                img_white[height_gap:height_calculated+height_gap, :] = image_resize
                prediction, indx = classifier.getPrediction(img_white, draw=False)
            
            cv2.rectangle(img_out, (x - OFFSET, y - OFFSET), (x + w+OFFSET, y + h+OFFSET), (139,0,0), 4)          # Rectangular box surrounding the detected hand
            cv2.putText(img_out, LABELS[indx], (x, y-OFFSET), cv2.FONT_HERSHEY_TRIPLEX, 2,(255, 255, 255), 2)       # The letter shown above the rectangualar box
            
            cv2.imshow("ImageCrop", img_crop)
            cv2.imshow("ImageWhite", img_white)
            
        cv2.imshow("Image", img_out)
        cv2.waitKey(1)

if __name__ == "__main__":
	main()
