import tensorflow as tf
from tensorflow.keras.datasets import mnist

class MNISTDataLoader:
    def __init__(self, config):
        self.batch_size = config["general"]["batch_size"]
        (self.train_images, _), (self.test_images, _) = mnist.load_data()
        self.train_images = (self.train_images.astype('float32') - 127.5) / 127.5
        self.train_images = tf.expand_dims(self.train_images, axis=-1)

    def get_train_data(self):
        train_dataset = tf.data.Dataset.from_tensor_slices(self.train_images)
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return train_dataset