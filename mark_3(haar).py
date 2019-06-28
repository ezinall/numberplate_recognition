import cv2 as cv

number_plate_cascade = cv.CascadeClassifier('haarcascade_russian_plate_number.xml')

img = cv.imread('numberplate.jpg')
gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

plates = number_plate_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in plates:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()
