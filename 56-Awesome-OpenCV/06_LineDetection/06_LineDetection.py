import os
import cv2
import numpy as np

video_path = r"56-Awesome-OpenCV\06_LineDetection\video01.mp4"
if not os.path.isfile(video_path):
    print("ERROR: Video file missing!")
    exit()
    
video = cv2.VideoCapture(video_path)

while True:
    success, frame = video.read()
    
    if not success:
        video = cv2.VideoCapture(video_path)
        continue
    
    gaussian_blured = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv_converted = cv2.cvtColor(gaussian_blured, cv2.COLOR_BGR2HSV)
    
    lower_yellow_bound = np.array([18, 94, 140])
    upper_yellow_bound = np.array([48, 255, 255])
    
    mask = cv2.inRange(hsv_converted, lower_yellow_bound, upper_yellow_bound)
    edges = cv2.Canny(mask, 74, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, maxLineGap=50)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
            
    cv2.imshow("Lane Detection", frame)
    cv2.imshow("Edges", edges)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
