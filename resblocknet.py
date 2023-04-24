from torch import nn
import brevitas.nn as qnn
from convblock import ConvBlock
import inspect
from util import initialise


class ResBlockNet(nn.Module):
    def __init__(self, config: dict, **kwargs):
        self.config = config
        argspec = inspect.getfullargspec(self.__init__)[0][1:]
        argdict = dict(zip(argspec, argspec))
        argdict.update(kwargs)
        self.args = argdict
        super(ResBlockNet, self).__init__()
        self.input_layer = ConvBlock(
            in_channels=config["input_layer"]["in_channels"],
            out_channels=config["input_layer"]["out_channels"],
            kernel_size=tuple(config["input_layer"]["kernel"]),
            type="conv",
        )
        self.map_blocks = nn.ModuleList()
        for i in range(len(config["mapping_blocks"])):
            self.map_blocks.append(
                nn.Sequential(
                    ConvBlock(
                        in_channels=config["mapping_blocks"][f"{i}"]["0"][
                            "in_channels"
                        ],
                        out_channels=config["mapping_blocks"][f"{i}"]["0"][
                            "out_channels"
                        ],
                        kernel_size=tuple(
                            config["mapping_blocks"][f"{i}"]["0"]["kernel"]
                        ),
                        type="dws",
                    ),
                    qnn.QuantReLU(),
                    ConvBlock(
                        in_channels=config["mapping_blocks"][f"{i}"]["1"][
                            "in_channels"
                        ],
                        out_channels=config["mapping_blocks"][f"{i}"]["1"][
                            "out_channels"
                        ],
                        kernel_size=tuple(
                            config["mapping_blocks"][f"{i}"]["0"]["kernel"]
                        ),
                        type="dws",
                    ),
                )
            )
        self.hidden_layers = nn.ModuleList()
        for i in range(len(config["hidden_layers"])):
            self.hidden_layers.append(
                ConvBlock(
                    in_channels=config["hidden_layers"][f"{i}"]["in_channels"],
                    out_channels=config["hidden_layers"][f"{i}"]["out_channels"],
                    kernel_size=tuple(config["hidden_layers"][f"{i}"]["kernel"]),
                    type="dws",
                )
            )
        self.output_layer = qnn.QuantConvTranspose2d(
            in_channels=config["output_layer"]["in_channels"],
            out_channels=config["output_layer"]["out_channels"],
            kernel_size=tuple(config["output_layer"]["kernel"]),
            padding=(config["output_layer"]["kernel"][0] // 2, config["output_layer"]["kernel"][1] // 2),
            stride=2,
            output_padding=1,
        )
        initialise(self.output_layer)

    def forward(self, out):
        out = self.input_layer(out)
        for map_block in self.map_blocks:
            residual = out
            out = map_block(self.activation(out))
            out = out + residual
        for hidden_layer in self.hidden_layers:
            out = hidden_layer(self.activation(out))
        out = self.output_layer(self.activation(out))
        return out


# expand after hidden layers
class ShrinkResBlockNet1(ResBlockNet):
    def __init__(self, config: dict, **kwargs):
        self.config = config
        super(ShrinkResBlockNet1, self).__init__()
        if "shrinking_layer" not in self.config:
            raise KeyError("ShrinkResBlockNet must have a shrinking layer")
        if "expanding_layer" not in self.config:
            raise KeyError("ShrinkResBlockNet must have an expanding layer")
        self.shrinking = ConvBlock(
            in_channels=config["shrinking_layer"]["in_channels"],
            out_channels=config["shrinking_layer"]["out_channels"],
            kernel_size=tuple(config["shrinking_layer"]["kernel"]),
            type="dws",
        )
        self.expanding = ConvBlock(
            in_channels=config["expanding_layer"]["in_channels"],
            out_channels=config["expanding_layer"]["out_channels"],
            kernel_size=tuple(config["expanding_layer"]["kernel"]),
            type="dws",
        )

    def forward(self, out):
        out = self.input_layer(out)
        out = self.shrinking(self.activation(out))
        for map_block in self.map_blocks:
            residual = out
            out = map_block(self.activation(out))
            out = out + residual
        for hidden_layer in self.hidden_layers:
            out = hidden_layer(self.activation(out))
        out = self.expanding(self.activation(out))
        out = self.output_layer(self.activation(out))
        return out


# no expand
class ShrinkResBlockNet2(ResBlockNet):
    def __init__(self, config: dict, **kwargs):
        self.config = config
        super(ShrinkResBlockNet1, self).__init__()
        if "shrinking_layer" not in self.config:
            raise KeyError("ShrinkResBlockNet must have a shrinking layer")
        self.shrinking = ConvBlock(
            in_channels=config["shrinking_layer"]["in_channels"],
            out_channels=config["shrinking_layer"]["out_channels"],
            kernel_size=tuple(config["shrinking_layer"]["kernel"]),
            type="dws",
        )

    def forward(self, out):
        out = self.input_layer(out)
        out = self.shrinking(self.activation(out))
        for map_block in self.map_blocks:
            residual = out
            out = map_block(self.activation(out))
            out = out + residual
        #out = self.expanding(self.activation(out))
        for hidden_layer in self.hidden_layers:
            out = hidden_layer(self.activation(out))
        out = self.output_layer(self.activation(out))
        return out
