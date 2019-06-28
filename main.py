import random

import cv2 as cv
import tensorflow as tf

# from model_v1 import model

from dataset import load_data


train_images, train_labels, test_images, test_labels = load_data()


def test_model_v1(model, frame):
    frame = cv.resize(frame, (128, 64))

    # frame = test_images[random.randrange(100)]
    # gray = frame

    height, width, *_ = frame.shape

    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY) / 255

    tmp = frame.copy()

    predict = model.predict([[gray]])
    predict_img = (predict[0] + 1) * (64, 32, 64, 32)

    left = int(width / 128 * predict_img[0])
    top = int(height / 64 * predict_img[1])
    right = int(width / 128 * predict_img[2])
    bottom = int(height / 64 * predict_img[3])

    cv.rectangle(tmp, (left, top), (right, bottom), (0, 255, 0), 1)
    cv.imshow('tmp', tmp)

    cv.waitKey(0)


def sliding_window(image, step_size, window_size=(128, 64)):
    for i in range(0, image.shape[0], step_size):
        for k in range(0, image.shape[1], step_size):
            yield (k, i, image[i:i + window_size[1], k:k + window_size[0]])


if __name__ == '__main__':
    model = tf.keras.models.load_model('model_v1.h5')

    model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    results = model.evaluate(test_images, test_labels)

    img = cv.imread('numberplate.jpg')
    img = cv.resize(img, (848, 480))
    cv.imshow('img', img)
    cv.waitKey(0)

    w, h = int(128 * 1), int(64 * 1)

    sliding = sliding_window(image=img, step_size=32, window_size=(w, h))

    for (x, y, window) in sliding:
        # clone = img.copy()
        # cv.rectangle(clone, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # if window.shape == (64, 128, 3):
        test_model_v1(model, window)

        # cv.imshow('clone', clone)
        # cv.imshow('window', window)

        # cv.waitKey(0)

