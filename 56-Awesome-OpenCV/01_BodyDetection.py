import os
import cv2

xml_file_path = r"56-Awesome-OpenCV\haarcascade_fullbody.xml"
if not os.path.isfile(xml_file_path):
    print("ERROR: XML file missing!")
    exit()

body_detector = cv2.CascadeClassifier(xml_file_path)

img = cv2.imread(r"56-Awesome-OpenCV\picture01.jpg")

bodies = body_detector.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)

if len(bodies) > 0:
    for (x, y, w, h) in bodies:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
else:
    print("No bodies detected. Try different parameters or cascade file")
    
cv2.imshow('frame', img)

cv2.waitKey(0)    
cv2.destroyAllWindows()
