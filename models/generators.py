import tensorflow as tf
from tensorflow.keras import layers

class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.model = tf.keras.Sequential()
        for layer_config in config["generator"]["layers"]:
            if layer_config["type"] == "conv_transpose":
                self.model.add(layers.Conv2DTranspose(
                    filters=layer_config["out_channels"],
                    kernel_size=layer_config["kernel_size"],
                    strides=layer_config["stride"],
                    padding='same'))
                if layer_config["activation"] == "ReLU":
                    self.model.add(layers.ReLU())
                elif layer_config["activation"] == "Tanh":
                    self.model.add(layers.Activation('tanh'))

    def call(self, inputs, training=None):
        output = self.model(inputs)
        # Assicurati che l'output abbia la stessa dimensione dell'input
        output = tf.image.resize(output, [28, 28])
        return output

    def print_model(self):
        self.model.build(input_shape=(None, 28, 28, 1))
        self.model.summary()