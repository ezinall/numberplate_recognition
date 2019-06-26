import tensorflow as tf

from dataset import load_data


DIGITS = "0123456789"
LETTERS = "ABCKEHMOPTXY"
CHARS = LETTERS + DIGITS


def get_loss(y, y_):
    # Calculate the loss from digits being incorrect.  Don't count loss from
    # digits that are in non-present plates.
    digits_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                                          tf.reshape(y[:, 1:],
                                                     [-1, len(CHARS)]),
                                          tf.reshape(y_[:, 1:],
                                                     [-1, len(CHARS)]))
    digits_loss = tf.reshape(digits_loss, [-1, 7])
    digits_loss = tf.reduce_sum(digits_loss, 1)
    digits_loss *= (y_[:, 0] != 0)
    digits_loss = tf.reduce_sum(digits_loss)

    # Calculate the loss from presence indicator being wrong.
    presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                                                          y[:, :1], y_[:, :1])
    presence_loss = 7 * tf.reduce_sum(presence_loss)

    return digits_loss, presence_loss, digits_loss + presence_loss


train_images, train_labels, test_images, test_labels = load_data()

print("check shapes:")
print("train_images - ", train_images.shape)
print("train_labels - ", train_labels.shape)
print("test_images - ", test_images.shape)
print("test_labels - ", test_labels.shape)

train_labels = train_labels / (64.0, 32.0, 64.0, 32.0) - 1.0
test_labels = test_labels / (64.0, 32.0, 64.0, 32.0) - 1.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Reshape([4, 64, 128, 1]),
    tf.keras.layers.Conv2D(48, (2, 2), input_shape=(5, 5, 1, 48), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(strides=(2, 2), padding='same'),

    tf.keras.layers.Conv2D(64, (2, 2), input_shape=(5, 5, 48, 64), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='same'),

    tf.keras.layers.Conv2D(128, (2, 2), input_shape=(5, 5, 64, 128), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(strides=(2, 2), padding='same'),

    tf.keras.layers.Reshape([8, 32, 128, 2048]),
    tf.keras.layers.Conv2D(2048, (2, 2), input_shape=(8 * 32 * 128, 2048), activation=tf.nn.relu),

    tf.keras.layers.Reshape([1, 1, 2048, 1 + 7 * len(CHARS)]),
    tf.keras.layers.Conv2D(1 + 7 * len(CHARS), (2, 2), input_shape=(2048, 1 + 7 * len(CHARS))),

    tf.keras.layers.Reshape([-1, 32 * 8 * 128]),
    tf.keras.layers.Dense(2048, input_shape=(32 * 8 * 128, 2048), activation=tf.nn.relu),

    tf.keras.layers.Dense(1 + 7 * len(CHARS), input_shape=(2048, 1 + 7 * len(CHARS))),
])

# model = tf.keras.models.load_model('model_v2.h5')


y = tf.keras.layers.Dense(1 + 7 * len(CHARS), input_shape=(2048, 1 + 7 * len(CHARS)))
y_ = tf.placeholder(tf.float32, [None, 7 * len(CHARS) + 1])
digits_loss, presence_loss, loss = get_loss(y, y_)

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001),
              loss=loss,
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=1, batch_size=50, verbose=True)

# model.save('model_v2.h5')

results = model.evaluate(test_images, test_labels)
