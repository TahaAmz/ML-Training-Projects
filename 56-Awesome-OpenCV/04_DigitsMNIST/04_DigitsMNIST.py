import cv2
import numpy as np

digits_train_path = r"56-Awesome-OpenCV\04_DigitsMNIST\DigitsTrain.png"
digits_train = cv2.imread(digits_train_path, cv2.IMREAD_GRAYSCALE)

digits_test_path = r"56-Awesome-OpenCV\04_DigitsMNIST\DigitsTest.png"
digits_test = cv2.imread(digits_test_path, cv2.IMREAD_GRAYSCALE)

digits_test = np.vsplit(digits_test, 50)
digits_test_data = []
for digit in digits_test:
    flattened = digit.flatten()
    digits_test_data.append(flattened)
digits_test_data = np.array(digits_test_data, dtype=np.float32)

digits_train = np.vsplit(digits_train, 50)
digits_train_data = []
for row in digits_train:
    digits_train_row = np.hsplit(row, 50)
    for digit in digits_train_row:
        flattened = digit.flatten()
        digits_train_data.append(flattened)
digits_train_data = np.array(digits_train_data, dtype=np.float32)

labels = np.repeat(range(10), 250)

knn = cv2.ml.KNearest_create() # type: ignore
knn.train(digits_train_data, cv2.ml.ROW_SAMPLE, labels)
    
ret, results, neighbours, dist = knn.findNearest(digits_test_data, k=3)

print("Results: ", results)
