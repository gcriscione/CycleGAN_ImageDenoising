import io
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras import layers
import logging

# Set up the logger
logging.basicConfig(filename='result/logs/models.log', level=logging.INFO, filemode='w')
logger = logging.getLogger(__name__)


class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()

        # Set seed for reproducibility
        seed = config['general'].get('seed', -1)
        if seed is not None and seed != -1:
            tf.random.set_seed(seed)

        # Create the generator model
        self.model = tf.keras.Sequential()
        for layer_config in config["generator"]["layers"]:
            self.add_layer(layer_config)

    # Add a layer to the generator model based on the configuration.
    def add_layer(self, layer_config):
        if layer_config["type"] == "conv_transpose":
            self.model.add(layers.Conv2DTranspose(
                filters=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                strides=layer_config["stride"],
                padding='same'))
            logger.info(f"Added Conv2DTranspose layer with {layer_config['out_channels']} filters")
        else:
            logger.warning(f"Unknown layer type: {layer_config['type']}")

        if "activation" in layer_config:
            if layer_config["activation"] == "ReLU":
                self.model.add(layers.ReLU())
                logger.info("Added ReLU activation")
            elif layer_config["activation"] == "Tanh":
                self.model.add(layers.Activation('tanh'))
                logger.info("Added Tanh activation")

        if "batch_norm" in layer_config and layer_config["batch_norm"]:
            self.model.add(layers.BatchNormalization())
            logger.info("Added BatchNormalization")

        if "dropout" in layer_config:
            self.model.add(layers.Dropout(layer_config["dropout"]))
            logger.info(f"Added Dropout with rate {layer_config['dropout']}")


    # Forward pass for the generator.
    def call(self, inputs, training=None):
        output = self.model(inputs)
        # Ensure output has the same size as the input
        output = tf.image.resize(output, [28, 28])
        return output

    # Print the model summary.
    def __str__(self):
        self.model.build(input_shape=(None, 28, 28, 1))
        with io.StringIO() as buf, redirect_stdout(buf):
            self.model.summary()
            model_summary = buf.getvalue()
        return model_summary