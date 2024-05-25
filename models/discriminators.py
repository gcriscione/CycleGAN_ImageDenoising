import io
from contextlib import redirect_stdout
import tensorflow as tf
from tensorflow.keras import layers
import logging

# Set up the logger
logging.basicConfig(filename='result/logs/models.log', level=logging.INFO, filemode='w')
logger = logging.getLogger(__name__)

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        
        # Set seed for reproducibility
        seed = config['general'].get('seed', -1)
        if seed is not None and seed != -1:
            tf.random.set_seed(seed)

        # Create the discriminator model
        self.model = tf.keras.Sequential()
        for layer_config in config["discriminator"]["layers"]:
            self.add_layer(layer_config)

    # Add a layer to the discriminator model based on the configuration.
    def add_layer(self, layer_config):
        if layer_config["type"] == "conv":
            self.model.add(layers.Conv2D(
                filters=layer_config["out_channels"],
                kernel_size=layer_config["kernel_size"],
                strides=layer_config["stride"],
                padding='same'))
            logger.info(f"Added Conv2D layer with {layer_config['out_channels']} filters")
        else:
            logger.warning(f"Unknown layer type: {layer_config['type']}")

        if "activation" in layer_config:
            if layer_config["activation"] == "LeakyReLU":
                self.model.add(layers.LeakyReLU())
                logger.info("Added LeakyReLU activation")
            elif layer_config["activation"] == "Sigmoid":
                self.model.add(layers.Activation('sigmoid'))
                logger.info("Added Sigmoid activation")

        if "batch_norm" in layer_config and layer_config["batch_norm"]:
            self.model.add(layers.BatchNormalization())
            logger.info("Added BatchNormalization")

        if "dropout" in layer_config:
            self.model.add(layers.Dropout(layer_config["dropout"]))
            logger.info(f"Added Dropout with rate {layer_config['dropout']}")

    # Forward pass for the discriminator.
    def call(self, inputs, training=None):
        return self.model(inputs)
    
    # Print the model summary.
    def __str__(self):
        self.model.build(input_shape=(None, 28, 28, 1))
        with io.StringIO() as buf, redirect_stdout(buf):
            self.model.summary()
            model_summary = buf.getvalue()
        return model_summary