from torch import nn
import brevitas.nn as qnn
from prefabs import ConvBlock
import inspect
from utils import initialise

# Basic architecture for FSRCNN, with compression layers


class ShrinkNet(nn.Module):
    def __init__(
        self,
        feature_channels=56,
        shrinking_channels=12,
        mapping_depth=4,
        types=["conv", "conv", "conv", "conv"],
        kernels=((5, 5), (5, 5), (3, 3), (1, 1), (9, 9)),
        **kwargs
    ):
        argspec = inspect.getfullargspec(self.__init__)[0][1:]
        argdict = dict(zip(argspec, argspec))
        argdict.update(kwargs)
        self.args = argdict
        super(ShrinkNet, self).__init__()

        if len(kernels) != 2 + mapping_depth + 2:
            raise ValueError("Not Enough Kernels")
        
        self.channels = 1
        self.feature_channels = feature_channels
        self.shrinking_channels = shrinking_channels
        self.mapping_depth = mapping_depth
        self.types = types
        self.kernels = kernels

        # First Part of FSRCNN, the Feature Extraction
        self.feature_extraction = ConvBlock(
            self.channels, self.feature_channels, kernel_size=self.kernels[0], type=self.types[0]
        )

        # Shrinking
        self.shrinking = ConvBlock(
            in_channels=self.feature_channels,
            out_channels=self.shrinking_channels,
            kernel_size=self.kernels[1],
            type=self.types[1],
        )

        # Mapping (Residual Layer)
        self.mapping = self.__generate_mapping(
            shrinking_channels=self.shrinking_channels,
            mapping_depth=self.mapping_depth,
            type=self.types[2],
            kernels=self.kernels[2:-2]
        )

        # Expanding
        self.expanding = ConvBlock(
            in_channels=self.shrinking_channels,
            out_channels=self.feature_channels,
            kernel_size=self.kernels[3],
            type=self.types[3],
        )

        # Deconv
        self.deconvolution = initialise(qnn.QuantConvTranspose2d(
            in_channels=self.feature_channels,
            out_channels=self.channels,
            kernel_size=self.kernels[4],
            padding=self.kernels[4][0] // 2,
            stride=2,
            output_padding=1,
        ))



        # Generate mapping layer

    def __generate_mapping(self, shrinking_channels, mapping_depth, type, kernels):
        to_return = nn.ModuleList()
        for i in range(mapping_depth):
            to_return.append(
                ConvBlock(
                    in_channels=shrinking_channels,
                    out_channels=shrinking_channels,
                    kernel_size=kernels[i],
                    type=type,
                )
            )
        return to_return

    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out


# ShrinkNet_Residual1:
# ShrinkNet_Residual1 makes the Mapping block a single residual block with a connection bypassing the whole mapping block
class ShrinkNet_Residual1(ShrinkNet):
    # FSRCNN_ResNet with variable Residual Mapping Layers.
    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        out = out + self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out


# adds a skip after every mapping layer, skipping the rest of the mapping layers
class ShrinkNet_Residual2(ShrinkNet):
    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        for mapping_layer in self.mapping:
            residual = out
            out = mapping_layer(out)
            out = out + residual
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out


# adds residual every other mapping layer
class ShrinkNet_Residual3(ShrinkNet):
    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        for i, mapping_layer in enumerate(self.mapping):
            residual = out
            out = mapping_layer(out)
            if i % 2 == 0:
                out = out + residual
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out


# skip to the end of mapping after every map layer
class ShrinkNet_Residual4(ShrinkNet):
    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        skip = out
        out = self.mapping[0](out)
        out = self.mapping[1](out) + skip
        out = self.mapping[2](out) + skip
        out = self.mapping[3](out) + skip
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out
