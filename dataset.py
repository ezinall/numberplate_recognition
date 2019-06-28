import os
import json

import cv2 as cv
import numpy as np


IMG_PATH = 'artificial_samples/artificial/img'
DATA_PATH = 'artificial_samples/artificial/ann'


def load_data(img_path=IMG_PATH, data_path=DATA_PATH):
    train_x, train_y, test_x, test_y = [], [], [], []

    for file in sorted(os.listdir(data_path)):
        with open(os.path.join(data_path, file)) as f:
            data = json.load(f)
            left = data["objects"][0]["points"]["exterior"][0][0]
            top = data["objects"][0]["points"]["exterior"][0][1]
            right = data["objects"][0]["points"]["exterior"][1][0]
            bottom = data["objects"][0]["points"]["exterior"][1][1]
            # bottom = data["objects"][0]["points"]["exterior"][2][1]

        if data["tags"][0] == 'train':
            train_y.append(np.array([left, top, right, bottom], dtype='int32'))

            img = cv.imread(os.path.join(img_path, file[:file.rfind('.')]))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255
            train_x.append(gray)
        else:
            test_y.append(np.array([left, top, right, bottom], dtype='int32'))

            img = cv.imread(os.path.join(img_path, file[:file.rfind('.')]))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255
            test_x.append(gray)

    train_x, test_x = np.array(train_x, dtype='float32'), np.array(test_x, dtype='float32')
    train_y, test_y = np.array(train_y, dtype='int32'), np.array(test_y, dtype='int32')

    return train_x, train_y, test_x, test_y
