import tensorflow as tf
import cv2 as cv

from dataset import load_data


def test_model(t_model, img):
    img = cv.resize(img, (128, 64))
    height, width, *_ = img.shape

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY) / 255

    tmp = img.copy()

    predict = t_model.predict([[gray]])
    predict_img = (predict[0] + 1) * (64, 32, 64, 32)
    print(predict[0])
    print(predict_img)

    left = int(width / 128 * predict_img[0])
    top = int(height / 64 * predict_img[1])
    right = int(width / 128 * predict_img[2])
    bottom = int(height / 64 * predict_img[3])

    cv.rectangle(tmp, (left, top), (right, bottom), (0, 255, 0), 1)
    cv.imshow('tmp', tmp)

    cv.waitKey(0)


train_images, train_labels, test_images, test_labels = load_data()

print("check shapes:")
print("train_images - ", train_images.shape)
print("train_labels - ", train_labels.shape)
print("test_images - ", test_images.shape)
print("test_labels - ", test_labels.shape)

train_labels = train_labels / (64.0, 32.0, 64.0, 32.0) - 1.0
test_labels = test_labels / (64.0, 32.0, 64.0, 32.0) - 1.0


model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape([64, 128, 1]),

    tf.keras.layers.Conv2D(32, (2, 2), input_shape=(3, 3, 1, 32), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(padding='same'),

    tf.keras.layers.Conv2D(64, (2, 2), input_shape=(2, 2, 32, 64), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(padding='same'),

    tf.keras.layers.Conv2D(128, (2, 2), input_shape=(2, 2, 64, 128), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(padding='same'),

    tf.keras.layers.Reshape([8 * 16 * 128]),

    tf.keras.layers.Dense(500, input_shape=(8 * 16 * 128, 500), activation=tf.nn.relu),
    tf.keras.layers.Dense(500, input_shape=(500, 500), activation=tf.nn.relu),
    tf.keras.layers.Dense(4, input_shape=(500, 4)),
])

model = tf.keras.models.load_model('model_v1.h5')

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['accuracy'])

# model.summary()

# history = model.fit(train_images, train_labels, epochs=30, batch_size=20, verbose=True)

model.save('model_v1.h5')

results = model.evaluate(test_images, test_labels)


img = cv.imread('numberplate_3.jpg')
test_model(model, img)
