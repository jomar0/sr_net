from torch import nn
import brevitas.nn as qnn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, type='conv'):
        super(ConvBlock, self).__init__()
        if (type == 'conv'):
            self.conv = qnn.QuantConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size[0]//2
            )
        elif type == 'dws':
            self.conv = nn.Sequential(
                qnn.QuantConv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=kernel_size,
                    groups=in_channels,
                    padding=kernel_size[0]//2
                ),
                qnn.QuantConv2d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=(
                        1, 1)
                )
            )
        self.activation = qnn.QuantReLU()
        self.initialise_weights(type=type)

    def initialise_weights(self, type='conv'):
        if (type == 'conv'):
            # Initialise Weights
            nn.init.kaiming_normal_(
                self.conv.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.conv.bias)

        elif type == 'dws':
            # Initalise DW weights
            nn.init.kaiming_normal_(
                self.conv[0].weight, mode='fan_in', nonlinearity='conv2d')
            nn.init.zeros_(self.conv[0].bias)

            # Initalise PW weights
            nn.init.kaiming_normal_(
                self.conv[1].weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(self.conv[1].bias)

    def forward(self, output):
        output = self.conv(output)
        output = self.activation(output)
        return output
