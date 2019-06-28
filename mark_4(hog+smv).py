import cv2 as cv
import numpy as np

# train:
hog = cv.HOGDescriptor()

train_list = []
response_list = []
for i in range(2, 7):
    img = cv.imread(str(i) + "_.png", cv.IMREAD_GRAYSCALE)
    h = hog.compute(img)
    train_list.append(h)
    response_list.append(i)

model = cv.SVM()
x = model.train(np.array(train_list), np.array(response_list))
model.save('trained.xml')


# predict:
svm = cv.SVM()
svm.load('trained.xml')

img = cv.imread('2.jpg', cv.IMREAD_GRAYSCALE)
h = hog.compute(img)
p = svm.predict(h)
