import cv2 as cv

from model_v2 import model


def test_model(model):
    img = cv.imread('numberplate.jpg')
    # img = test_images[91]
    height, width, *_ = img.shape

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, (128, 64))
    # gray = img

    predict = model.predict([[gray / 255]])
    predict_img = (predict[0] + 1) * (64, 32, 64, 32)

    left = int(width / 128 * predict_img[0])
    top = int(height / 64 * predict_img[1])
    right = int(width / 128 * predict_img[2])
    bottom = int(height / 64 * predict_img[3])

    cv.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 1)
    cv.imshow('img', img)

    cv.waitKey(0)


if __name__ == '__main__':
    test_model(model)
