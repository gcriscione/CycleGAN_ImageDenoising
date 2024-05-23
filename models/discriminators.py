import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        layers = []
        input_channels = config['input_channels']

        for layer_cfg in config['layers']:
            if layer_cfg['type'] == 'conv':
                layers.append(
                    nn.Conv2d(
                        in_channels=input_channels,
                        out_channels=layer_cfg['out_channels'],
                        kernel_size=layer_cfg['kernel_size'],
                        stride=layer_cfg['stride'],
                        padding=layer_cfg['padding'],
                        bias=False
                    )
                )
                if layer_cfg['batch_normalization']:
                    layers.append(nn.InstanceNorm2d(layer_cfg['out_channels'], affine=True))
                if layer_cfg['activation'] == 'LeakyReLU':
                    layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif layer_cfg['activation'] == 'Sigmoid':
                    layers.append(nn.Sigmoid())
                if layer_cfg['dropout']:
                    layers.append(nn.Dropout(0.5))

                input_channels = layer_cfg['out_channels']

        self.model = nn.Sequential(*layers)
        self._initialize_weights(config['layers'])

    def _initialize_weights(self, layers_config):
        for layer_cfg, layer in zip(layers_config, self.model):
            if isinstance(layer, (nn.ConvTranspose2d, nn.Conv2d)):
                if layer_cfg['weight_initialization'] == 'xavier':
                    nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.model(x)
    
    def __str__(self):
        return f"Discriminator Model:\n{self.model}"