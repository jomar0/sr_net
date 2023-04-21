from torch import nn
import brevitas.nn as qnn
from prefabs import ConvBlock
import inspect


class ResBlockNet(nn.Module):
    def __init__(self, config: dict, **kwargs):
        argspec = inspect.getfullargspec(self.__init__)[0][1:]
        argdict = dict(zip(argspec, argspec))
        argdict.update(kwargs)
        self.args = argdict
        super(ResBlockNet, self)
        self.input_layer = ConvBlock(
            in_channels=config["input_layer"]["in_channels"],
            out_channels=config["input_layer"]["out_channels"],
            kernel_size=config["input_layer"]["kernel"],
            type="conv",
        )
        self.map_blocks = nn.ModuleList()
        for i in range(len(config["mapping_blocks"])):
            self.map_blocks.append(
                nn.Sequential(
                    ConvBlock(
                        in_channels=config["mapping_blocks"][i][0]["in_channels"],
                        out_channels=config["mapping_blocks"][i][0]["out_channels"],
                        kernel_size=config["mapping_blocks"][i][0]["kernel"],
                        type="dws",
                    ),
                    ConvBlock(
                        in_channels=config["mapping_blocks"][i][1]["in_channels"],
                        out_channels=config["mapping_blocks"][i][1]["out_channels"],
                        kernel_size=config["mapping_blocks"][i][0]["kernel"],
                        type="dws",
                    ),
                )
            )
        self.hidden_layers = nn.ModuleList()
        for i in range(len(config["hidden_layers"])):
            self.hidden_layers.append(ConvBlock(
                in_channels=config["hidden_layers"][i]["in_channels"],
                out_channels = config["hidden_layers"][i]["out_channels"],
                kernel_size=config["hidden_layers"][i]["kernel"],
                type="dws"
            ))
        self.output_layer = qnn.QuantConvTranspose2d(
            in_channels=config["output_layer"]["in_channels"],
            out_channels=config["output_layer"]["out_channels"],
            kernel_size=config["output_layer"]["kernel"],
            padding=config["output_layer"]["kernel"][0] // 2,
            stride=2,
            output_padding=1,
        )
        __initialise(self.output_layer)

        def __initialise(self, module):
            nn.init.xavier_normal_(module.weight)
            return module
    #def forward(self, out):
    