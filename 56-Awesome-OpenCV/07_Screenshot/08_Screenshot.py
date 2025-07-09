import cv2
import numpy as np
import pyautogui

img = pyautogui.screenshot()
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

cv2.imwrite("screenshot.png",img)
