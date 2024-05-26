# CycleGAN Image Denoising with MNIST

## Project Overview
This project aims to implement a CycleGAN for image denoising using the MNIST dataset. The project is organized into several modules with configurable settings for easy experimentation and extension.

## File Structure
- `config/`
  - `config.json`: Configuration file for setting model parameters, training settings, and noise types.
- `models/`
  - `generators.py`: Defines the `ResnetGenerator` class with ResNet blocks.
  - `discriminators.py`: Defines the `NLayerDiscriminator` class.
  - `losses.py`: Defines the loss functions used for training.
- `utils/`
  - `dataset.py`: Contains `MNISTDataLoader` for loading the dataset.
  - `noise.py`: Contains `NoiseAdder` for adding noise to the images.
  - `plot.py`: Functions for plotting images.
  - `stats.py`: Functions for printing training statistics.
  - `validate.py`: Script for validating the model during training.
  - `test.py`: Script for testing the model after training.
- `train.py`: Main script for training the models.
- `main.py`: Entry point for starting the training or testing based on the configuration.

## Configuration Guide

### General Settings
- `batch_size`: Number of samples per gradient update. 
- `num_epochs`: Number of epochs to train the model.
- `learning_rate`: Learning rate for the optimizer.
- `beta1`, `beta2`: Coefficients used for computing running averages of gradient and its square in the Adam optimizer.
- `data_path`: Path to the dataset.

### Generator Settings
- `input_dim`: Dimension of the input noise vector.
- `layers`: List of layers defining the generator architecture.
  - `type`: Layer type (`conv_transpose`).
  - `out_channels`: Number of output channels.
  - `kernel_size`: Size of the convolutional kernel.
  - `stride`: Stride of the convolution.
  - `padding`: Padding added to all four sides of the input.
  - `activation`: Activation function (`ReLU`, `Tanh`).

**Example:**
```json
"generator": {
    "input_dim": 1,
    "layers": [
        {
            "type": "conv_transpose",
            "out_channels": 32,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "activation": "ReLU"
        },
        {
            "type": "conv_transpose",
            "out_channels": 16,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "activation": "ReLU"
        },
        {
            "type": "conv_transpose",
            "out_channels": 1,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "activation": "Tanh"
        }
    ]
}
```

### Discriminator Settings
- `input_channels`: Number of input channels.
- `layers`: List of layers defining the discriminator architecture.
  - `type`: Layer type (`conv`).
  - `out_channels`: Number of output channels.
  - `kernel_size`: Size of the convolutional kernel.
  - `stride`: Stride of the convolution.
  - `padding`: Padding added to all four sides of the input.
  - `activation`: Activation function (`LeakyReLU`, `Sigmoid`).

**Example:**
```json
"discriminator": {
    "input_channels": 1,
    "layers": [
        {
            "type": "conv",
            "out_channels": 64,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "activation": "LeakyReLU"
        },
        {
            "type": "conv",
            "out_channels": 1,
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "activation": "Sigmoid"
        }
    ]
}
```

### Noise Adder Settings
- `noise_type`(string): Type of noise to add (`salt_and_pepper`, `gaussian`, `speckle`, `poisson`; default: `salt_and_pepper`).
- `salt_pepper_ratio`(float): Ratio of salt and pepper noise (default: 0.02).
- `gaussian_mean` (float): Mean of Gaussian noise (default: 0).
- `gaussian_std` (float): Standard deviation of Gaussian noise (default: 0.02).

**Example:**
```json
"noise_adder": {
    "noise_type": "salt_and_pepper",
    "salt_pepper_ratio": 0.02,
    "gaussian_mean": 0,
    "gaussian_std": 0.02
}
```

### Training Settings
- `loss_function`: Loss function to use (`BCE`, `MSE`).
- `optimizer`: Optimizer to use (`Adam`).
- `gradient_clipping`: Whether to apply gradient clipping.
  - Example: `false`

**Example:**
```json
"training": {
    "loss_function": "BCE",
    "optimizer": "Adam",
    "gradient_clipping": false
}
```

## Running the Project
### Python Version
Ensure you are using Python 3.9 to be compatible with the version of TensorFlow used in this project, which allows for GPU utilization. For detailed instructions on setting up TensorFlow with GPU support, visit: [TensorFlow Installation Guide](https://www.tensorflow.org/install/pip?hl=it#windows-native_1)

1. Install the required dependencies:
   ```bash
   pip install -r requirements
   ```
2. Run the training script:
   ```bash
   python main.py
   ```

3. Monitor the output and adjust configurations as needed.