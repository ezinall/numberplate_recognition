import tensorflow as tf

from dataset import load_data


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

# model = tf.keras.models.load_model('model_v1.h5')

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=30, batch_size=20, verbose=True)

# model.save('model_v1.h5')

results = model.evaluate(test_images, test_labels)
