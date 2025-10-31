# Importing the Libraries Required

import cv2
import numpy as np
import os
from string import ascii_uppercase

# --- Configuration ---
mode = 'testingData'
directory = os.path.join('dataSet', mode)
minValue = 35

# Define all the folders (classes) we need: '0', 'A', 'B', ... 'Z'
FOLDERS = ['0'] + list(ascii_uppercase)

# --- Directory Setup (CRITICAL FIX for FileNotFoundError) ---
# Ensure the base directory and all subdirectories exist before listing files.
for folder in FOLDERS:
    path = os.path.join(directory, folder)
    # os.makedirs with exist_ok=True will create directories only if they don't exist
    os.makedirs(path, exist_ok=True)

capture = cv2.VideoCapture(0)
interrupt = -1

while True:
    _, frame = capture.read()

    # Simulating mirror Image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images (using FOLDERS list for robustness)
    count = {'zero': len(os.listdir(os.path.join(directory, "0")))}
    for char in ascii_uppercase:
        count[char.lower()] = len(os.listdir(os.path.join(directory, char)))

    # --- Print Counts on Screen ---
    y_offset = 60
    cv2.putText(frame, "ZERO : " + str(count['zero']), (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    y_offset += 10
    
    for char in ascii_uppercase:
        cv2.putText(frame, f"{char.lower()} : {count[char.lower()]}", (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        y_offset += 10

    # Coordinates of the ROI (Region of Interest)
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])

    # Drawing the ROI
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    
    # Extracting the ROI
    roi = frame[y1:y2, x1:x2]

    cv2.imshow("Frame", frame)
    
    # --- Image Processing ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Output Image after the Image Processing that is used for data collection
    test_image = cv2.resize(test_image, (300, 300))
    cv2.imshow("test", test_image)

    # --- Data Collection ---
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        # esc key
        break
        
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(os.path.join(directory, '0', f"{count['zero']}.jpg"), roi)

    for char in ascii_uppercase:
        if interrupt & 0xFF == ord(char.lower()):
            # Save the image to the corresponding uppercase folder (A, B, C, ...)
            cv2.imwrite(os.path.join(directory, char, f"{count[char.lower()]}.jpg"), roi)
            break
            
capture.release()
cv2.destroyAllWindows()
