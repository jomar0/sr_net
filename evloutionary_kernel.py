from torch import nn
import brevitas.nn as qnn
from prefabs import ConvBlock


class EVNet_Base(nn.Module):
    def __init__(self, kernels, channels):
        super(EVNet_Base, self)
        self.layers = list([])
        for i in len(channels):
            self.layers.append(
                ConvBlock(
                    in_channels=1 if i == 0 else channels[i - 1],
                    out_channels=channels[i],
                    kernel_size=[i],
                    type="conv" if i == 0 else "dws",
                )
            )
        self.layers.append(
            qnn.QuantConvTranspose2d(
                in_channels=channels[len(channels) - 1],
                out_channels=1,
                kernel_size=kernels[len(kernels) - 1],
                padding=kernels[len(kernels) - 1][0] // 2,
                stride=2,
                output_padding=1,
            )
        )
    

