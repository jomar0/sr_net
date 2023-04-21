from torch import nn
import brevitas.nn as qnn
from prefabs import ConvBlock
import inspect


class EVNet(nn.Module):
    def __init__(self, kernels, channels, **kwargs):
        argspec = inspect.getfullargspec(self.__init__)[0][1:]
        argdict = dict(zip(argspec, argspec))
        argdict.update(kwargs)
        self.args = argdict

        super(EVNet, self)
        self.hidden_layers = nn.ModuleList()
        self.input_layer = ConvBlock(
            in_channels=1, out_channels=channels[0], kernel_size=kernels[0], type="conv"
        )
        for i in range(1, len(channels) -1):
            self.hidden_layers.append(
                ConvBlock(
                    in_channels=channels[i - 1],
                    out_channels=channels[i],
                    kernel_size=[i],
                    type="conv" if i == 0 else "dws",
                )
            )
        self.output_layer = qnn.QuantConvTranspose2d(
            in_channels=channels[len(channels) - 1],
            out_channels=1,
            kernel_size=kernels[len(kernels) - 1],
            padding=kernels[len(kernels) - 1][0] // 2,
            stride=2,
            output_padding=1,
        )
        __initialise(self.output_layer)

        def __initialise(self, module):
            nn.init.xavier_normal_(module.weight)
            return module
        def forward(self, out):
            out = self.input_layer(out)
            for hidden_layer in self.hidden_layers:
                out = hidden_layer(out)
            out = self.output_layer(out)
            return out