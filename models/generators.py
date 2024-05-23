import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        layers = []
        input_dim = config['input_dim']

        for idx, layer_cfg in enumerate(config['layers']):
            if layer_cfg['type'] == 'conv_transpose':
                layers.append(
                    nn.ConvTranspose2d(
                        in_channels=input_dim,
                        out_channels=layer_cfg['out_channels'],
                        kernel_size=layer_cfg['kernel_size'],
                        stride=layer_cfg['stride'],
                        padding=layer_cfg['padding'],
                        bias=False
                    )
                )
                if layer_cfg['batch_normalization'] and idx < len(config['layers']) - 1:
                    layers.append(nn.BatchNorm2d(layer_cfg['out_channels']))
                if layer_cfg['activation'] == 'ReLU':
                    layers.append(nn.ReLU(inplace=True))
                elif layer_cfg['activation'] == 'Tanh':
                    layers.append(nn.Tanh())
                if layer_cfg['dropout']:
                    layers.append(nn.Dropout(0.5))

                input_dim = layer_cfg['out_channels']

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
        return f"Generator Model:\n{self.model}"