import torch.nn as nn

# Generator model
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_channels, feature_maps, activation_function, batch_normalization, dropout, layer_type, kernel_size, stride, padding, weight_initialization):
        super(Generator, self).__init__()
        
        layers = []
        in_channels = input_dim

        for i in range(4):
            layers.append(self._create_layer(in_channels, feature_maps * 2 ** i, activation_function, batch_normalization, dropout, layer_type, kernel_size, stride, padding))
            in_channels = feature_maps * 2 ** i

        layers.append(nn.ConvTranspose2d(in_channels, output_channels, kernel_size, stride=1, padding=0))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)
        
        self._initialize_weights(weight_initialization)
    
    def _create_layer(self, in_channels, out_channels, activation_function, batch_normalization, dropout, layer_type, kernel_size, stride, padding):
        layers = []
        if layer_type == 'conv_transpose':
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        if batch_normalization:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if activation_function == 'ReLU':
            layers.append(nn.ReLU(True))
        elif activation_function == 'LeakyReLU':
            layers.append(nn.LeakyReLU(0.2, True))
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")

        if dropout:
            layers.append(nn.Dropout(0.5))

        return nn.Sequential(*layers)
    
    def _initialize_weights(self, initialization_type):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                if initialization_type == 'xavier':
                    nn.init.xavier_normal_(m.weight)
                elif initialization_type == 'normal':
                    nn.init.normal_(m.weight, 0.0, 0.02)
                else:
                    raise ValueError(f"Unsupported weight initialization: {initialization_type}")
    
    def forward(self, x):
        return self.model(x)

