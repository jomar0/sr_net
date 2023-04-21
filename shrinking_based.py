from torch import nn
import brevitas.nn as qnn
from prefabs import ConvBlock

# Basic architecture for FSRCNN, with compression layers


class Shrinking_BaseNet(nn.Module):
    def __init__(
        self,
        channels=1,
        feature_dimension=56,
        shrinking_filters=12,
        mapping_depth=4,
        types=["conv", "conv", "conv", "conv"],
        kernels=((5, 5), (5, 5), (3, 3), (1, 1), (9, 9)),
    ):
        super(Shrinking_BaseNet, self).__init__()

        self.channels = channels
        self.feature_dimension = feature_dimension
        self.shrinking_filters = shrinking_filters
        self.mapping_depth = mapping_depth
        self.types = types
        self.kernels = kernels

        # First Part of FSRCNN, the Feature Extraction
        self.feature_extraction = ConvBlock(
            self.channels, self.feature_dimension, kernel_size=self.kernels[0], type=self.types[0]
        )

        # Shrinking
        self.shrinking = ConvBlock(
            in_channels=self.feature_dimension,
            out_channels=self.shrinking_filters,
            kernel_size=self.kernels[1],
            type=self.types[1],
        )

        # Mapping (Residual Layer)
        self.mapping = self.__generate_mapping(
            shrinking_filters=self.shrinking_filters,
            mapping_depth=self.mapping_depth,
            type=self.types[2],
            kernel=self.kernels[2]
        )

        # Expanding
        self.expanding = ConvBlock(
            in_channels=self.shrinking_filters,
            out_channels=self.feature_dimension,
            kernel_size=self.kernels[3],
            type=self.types[3],
        )

        # Deconv
        self.deconvolution = self.__initialise(qnn.QuantConvTranspose2d(
            in_channels=self.feature_dimension,
            out_channels=self.channels,
            kernel_size=self.kernels[4],
            padding=self.kernels[4][0] // 2,
            stride=2,
            output_padding=1,
        ))

    def __initialise(self, module):
        nn.init.xavier_normal_(module.weight)
        return module

        # Generate mapping layer

    def __generate_mapping(self, shrinking_filters, mapping_depth, type, kernel):
        to_return = list([])
        for _ in range(mapping_depth):
            to_return.append(
                ConvBlock(
                    in_channels=shrinking_filters,
                    out_channels=shrinking_filters,
                    kernel_size=kernel,
                    type=type,
                )
            )
        return nn.Sequential(*to_return)


# FSRCNN_ResNet1:
# FSRCNN_ResNet1 makes the Mapping block a single residual block with a connection bypassing the whole mapping block
class FSRCNN_ResNet1(Shrinking_BaseNet):
    # FSRCNN_ResNet with variable Residual Mapping Layers.
    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        out = out + self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out




# FSRCNN with no residual connections


class FSRCNN(Shrinking_BaseNet):
    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        out = self.mapping(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out



# FSRCNN_ResNet2:
# FSRCNN_ResNet2 introduces a skip connection over each mapping layer in the mapping block individually

class FSRCNN_ResNet2(Shrinking_BaseNet):
    def __init__(
        self,
        channels=1,
        feature_dimension=56,
        shrinking_filters=12,
        mapping_depth=4,
        types=["conv", "conv", "conv", "conv"],
        kernels=((5, 5), (5, 5), (3, 3), (1, 1), (9, 9))
    ):
        super(FSRCNN_ResNet2, self).__init__(channels=channels, feature_dimension=feature_dimension, shrinking_filters=shrinking_filters, mapping_depth=mapping_depth, types=types, kernels=kernels)
        self.mapping = None 
        self.mapping1 = ConvBlock(
                    in_channels=shrinking_filters,
                    out_channels=shrinking_filters,
                    kernel_size=self.kernels[2],
                    type=self.types[2],
                )
        self.mapping2 = ConvBlock(
                    in_channels=shrinking_filters,
                    out_channels=shrinking_filters,
                    kernel_size=self.kernels[2],
                    type=self.types[2],
                )
        self.mapping3 = ConvBlock(
                    in_channels=shrinking_filters,
                    out_channels=shrinking_filters,
                    kernel_size=self.kernels[2],
                    type=self.types[2],
                )
        self.mapping4 = ConvBlock(
                    in_channels=shrinking_filters,
                    out_channels=shrinking_filters,
                    kernel_size=self.kernels[2],
                    type=self.types[2],
                )

    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        out = out + self.mapping1(out)
        out = out + self.mapping2(out)
        out = out + self.mapping3(out)
        out = out + self.mapping4(out)
        out = self.expanding(out)
        out = self.deconvolution(out)
        return out
    
# adds a skip after every mapping layer, skipping the rest of the mapping layers
class FSRCNN_ResNet3(FSRCNN_ResNet2):
    def forward(self, out):
        out = self.feature_extraction(out)
        out = self.shrinking(out)
        skip = out
        out = self.mapping1(out)
        out= self.mapping2(out) + skip
        out = self.mapping3(out) + skip
        out = self.mapping4(out) + skip

        out = self.expanding(out)
        out =self.deconvolution(out)
        return out
    
    
    