import os
import cv2

xml_file_path = r"56-Awesome-OpenCV\02_EyeDetection\haarcascade_eye.xml"
if not os.path.isfile(xml_file_path):
    print("ERROR: XML file missing!")
    exit()
    
eye_detector = cv2.CascadeClassifier(xml_file_path)

img_path = r"56-Awesome-OpenCV\02_EyeDetection\eye01.jpg"
if not os.path.isfile(img_path):
    print("ERROR: Image file missing!")
    exit()

img = cv2.imread(img_path)

eyes = eye_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=8)

if len(eyes) > 0:
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
else:
    print("No eyes detected.")

cv2.imwrite("eyes_detected.jpg", img)
cv2.imshow('frame', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
    