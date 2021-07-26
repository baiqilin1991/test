import tensorflow as tf
import datetime
import numpy as np

mnist = tf.keras.datasets.mnist

# (x_train, y_train),(x_test, y_test) = mnist.load_data()

TRAIN_N = 60000
x_train = np.random.randint(0,255, [TRAIN_N, 28,28], dtype=np.uint8)
y_train = np.random.randint(0,9, [TRAIN_N,])

TEST_N = 10000
x_test = np.random.randint(0,255, [TEST_N, 28,28], dtype=np.uint8)
y_test = np.random.randint(0,9, [TEST_N,])

x_train, x_test = x_train / 255.0, x_test / 255.0

def create_model():
      return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
      ])
    
model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=False, histogram_freq=1)

model.fit(x=x_train, 
          y=y_train, 
          epochs=50, 
          validation_data=(x_test, y_test), 
          callbacks=[tensorboard_callback])
