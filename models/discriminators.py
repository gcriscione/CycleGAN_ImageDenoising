import tensorflow as tf
from tensorflow.keras import layers

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.model = tf.keras.Sequential()
        for layer_config in config["discriminator"]["layers"]:
            if layer_config["type"] == "conv":
                self.model.add(layers.Conv2D(
                    filters=layer_config["out_channels"],
                    kernel_size=layer_config["kernel_size"],
                    strides=layer_config["stride"],
                    padding='same'))
                if layer_config["activation"] == "LeakyReLU":
                    self.model.add(layers.LeakyReLU())
                elif layer_config["activation"] == "Sigmoid":
                    self.model.add(layers.Activation('sigmoid'))

    def call(self, inputs, training=None):
        return self.model(inputs)
    
    def print_model(self):
        self.model.build(input_shape=(None, 28, 28, 1))
        self.model.summary()