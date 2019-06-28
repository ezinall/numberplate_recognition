import os
import json

import cv2 as cv
import numpy as np


IMG_PATH = 'artificial/img'
DATA_PATH = 'artificial/ann'


def load_data(img_path=IMG_PATH, data_path=DATA_PATH, split=9781):
    img_list = []
    data_list = []
    for file in sorted(os.listdir(img_path)):
        img = cv.imread(os.path.join(img_path, file))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) / 255
        img_list.append(gray)
    for file in sorted(os.listdir(data_path)):
        with open(os.path.join(data_path, file)) as f:
            data = json.load(f)
            left = data["objects"][0]["points"]["exterior"][0][0]
            top = data["objects"][0]["points"]["exterior"][0][1]
            right = data["objects"][0]["points"]["exterior"][1][0]
            # bottom = data["objects"][0]["points"]["exterior"][1][1]
            bottom = data["objects"][0]["points"]["exterior"][2][1]
        data_list.append([left, top, right, bottom])
    img_data, data_data = np.array(img_list, dtype='float32'), np.array(data_list, dtype='int32')
    return img_data[:split], data_data[:split], img_data[split:], data_data[split:]
