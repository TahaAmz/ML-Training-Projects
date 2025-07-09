import cv2
import mss
import numpy as np
from PIL import ImageGrab

with mss.mss() as sct:
    monitor = sct.monitors[1]
    screen_width = monitor["width"]
    screen_height = monitor["height"]

    fourcc = cv2.VideoWriter_fourcc(*'XVID') # type: ignore
    out = cv2.VideoWriter("recording.avi", fourcc, 30.0, (screen_width, screen_height))
    cv2.namedWindow("Screen Recording", cv2.WINDOW_NORMAL)

    while True:
        img = ImageGrab.grab()
        img_np = np.array(img)
        
        frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
        
        out.write(frame_bgr)
        
        cv2.imshow("Screen Recording", frame_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
out.release()
cv2.destroyAllWindows()