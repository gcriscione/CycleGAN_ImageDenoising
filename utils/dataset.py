import tensorflow as tf

class MNISTDataLoader:
    def __init__(self, config):
        self.config = config
        self.training_size = config['general'].get('training_size', None)
        self.validation_size = config['general'].get('validation_size', None)
        self.test_size = config['general'].get('test_size', None)
        self.img_size = config['general'].get('img_size', 28)

    def get_train_data(self):
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_train = x_train[..., tf.newaxis]  # Add a channel dimension

        if self.training_size is not None:
            x_train = x_train[:self.training_size]
            y_train = y_train[:self.training_size]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(60000).batch(self.config['general']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return train_dataset
    
    def get_validation_data(self):
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        x_val = x_train.astype('float32') / 255.0
        x_val = x_val[..., tf.newaxis]  # Add a channel dimension
        
        if self.validation_size is not None:
            x_val = x_val[:self.validation_size]
            y_train = y_train[:self.validation_size]
        
        validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_train))
        validation_dataset = validation_dataset.batch(self.config['general']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return validation_dataset
    
    def get_test_data(self):
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.astype('float32') / 255.0
        x_test = x_test[..., tf.newaxis]  # Add a channel dimension
        
        if self.test_size is not None:
            x_test = x_test[:self.test_size]
            y_test = y_test[:self.test_size]
        
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        test_dataset = test_dataset.batch(self.config['general']['batch_size']).prefetch(tf.data.experimental.AUTOTUNE)
        return test_dataset