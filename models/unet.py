""" UNet architecture """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """ Residual Block of https://arxiv.org/abs/1908.02182,
    implementation at https://github.com/MIC-DKFZ/nnUNet
    """

    def __init__(self, num_channels_in: int, num_channels_out: int, p_dropout: float = None):

        super().__init__()
        ConvLayer = nn.Conv3d
        norm = nn.InstanceNorm3d

        self.conv1 = ConvLayer(num_channels_in, num_channels_out,
                               kernel_size=3, padding=1)
        self.norm1 = norm(num_channels_out)
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = nn.Identity()

        self.conv2 = ConvLayer(
            num_channels_out, num_channels_out, kernel_size=3, padding=1
        )
        self.norm2 = norm(num_channels_out)

        # 1x1x1 conv to adapt channels of residual
        if num_channels_in != num_channels_out:
            self.adapt_skip = nn.Sequential(ConvLayer(num_channels_in,
                                                      num_channels_out, 1,
                                                      bias=False),
                                            norm(num_channels_out))
        else:
            self.adapt_skip = nn.Identity()

    def forward(self, x):
        # Conv --> Norm --> ReLU --> (Dropout)
        x_out = self.conv1(x)
        x_out = F.relu(self.norm1(x_out))
        x_out = self.dropout(x_out)
        x_out = self.norm2(self.conv2(x_out))
        res = self.adapt_skip(x)
        x_out += res

        return F.relu(x_out)


class Resize3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        return F.interpolate(x, size=size, mode='trilinear', align_corners=True)


class ResidualUNetEncoder(nn.Module):
    """ Residual UNet encoder oriented on https://github.com/MIC-DKFZ/nnUNet.

    :param input_channels: The number of image channels
    :param encoder_channels: List of channel dimensions of all feature maps
    :returns: Encoded feature maps for every encoder step
    """

    def __init__(self, input_channels: int, encoder_channels, p_dropout: float):
        super().__init__()

        self.num_steps = len(encoder_channels)
        self.channels = encoder_channels

        # Initial step: Conv --> Residual block
        self.first_layer = nn.Sequential(
            nn.Conv3d(input_channels, self.channels[0], 3, padding=1),
            ResidualBlock(self.channels[0], self.channels[0], p_dropout=p_dropout)
        )

        # main layers
        down_layers = [Resize3D() for _ in range(1, self.num_steps)]
        self.down_layers = nn.ModuleList(down_layers)

        # down layers
        layers = [ResidualBlock(self.channels[i - 1], self.channels[i], p_dropout=p_dropout)
                  for i in range(1, self.num_steps)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        skips = []

        x = self.first_layer(x)
        skips.append(x)
        for i in range(self.num_steps - 1):
            size = [s // 2 for s in x.size()[2:]]
            x = self.down_layers[i](x, size=size)
            x = self.layers[i](x)
            skips.append(x)

        return skips


class ResidualUNetDecoder(nn.Module):
    """ Residual UNet decoder oriented on https://github.com/MIC-DKFZ/nnUNet.

    :param encoder: The encoder from which the decoder receives features
    :param decoder_channels: List of channel dimensions of all feature maps
    :param num_classes: The number of classes to segment
    :returns: Segmentation output
    """

    def __init__(self, encoder, decoder_channels, num_classes,
                 deep_supervision, p_dropout, input_size, skips_only):
        super().__init__()
        # Decoder has one step less
        num_steps = encoder.num_steps - 1
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.channels = decoder_channels
        self.deep_supervision = deep_supervision
        self.deep_supervision_pos = (1, 2)  # デコーダの層を下層よりゼロから数えた番号
        self.input_size = input_size
        self.skips_only = skips_only

        self.up_layers = [Resize3D() for _ in range(num_steps)]
        self.main_layers = [ResidualBlock(encoder.channels[-2] + encoder.channels[-1],
                                          self.channels[0], p_dropout=p_dropout)]
        for i in range(1, self.num_steps):
            self.main_layers.append(
                ResidualBlock(encoder.channels[-i - 2] + self.channels[i - 1], self.channels[i], p_dropout=p_dropout)
            )

        self.up_layers = nn.ModuleList(self.up_layers)
        self.main_layers = nn.ModuleList(self.main_layers)

        if deep_supervision:
            dsv_layers = [
                nn.Conv3d(self.channels[i], num_classes, 1, bias=False) for i in range(1, num_steps)
                if deep_supervision and i in self.deep_supervision_pos
            ]
            self.dsv_layers = nn.ModuleList(dsv_layers)

        if not self.skips_only:
            # Segmentation layer
            self.final_layer = nn.Conv3d(self.channels[-1], num_classes, 1, bias=False)

    def forward(self, skips):
        _skips = skips[::-1]
        x = _skips[0]
        results = []
        up_skips = []

        j = 0
        for i in range(self.num_steps):
            x = self.up_layers[i](x, size=_skips[i + 1].size()[2:])
            x = torch.cat((x, _skips[i + 1]), dim=1)
            x = self.main_layers[i](x)
            up_skips.append(x)

            if self.deep_supervision and (self.num_steps - i - 1) in self.deep_supervision_pos:
                y = self.dsv_layers[j](x)
                y = F.interpolate(y, size=self.input_size, mode='trilinear', align_corners=True)
                results.append(y)
                j += 1

        if self.skips_only:
            return up_skips

        pred = self.final_layer(x)
        results.append(pred)
        return up_skips, results


class ResidualUNet(nn.Module):
    """ Residual UNet oriented on https://github.com/MIC-DKFZ/nnUNet.
    It allows to flexibly exchange the size of the decoder and to get feature
    maps from different stages of the encoder and/or decoder.
    """

    def __init__(self, num_classes: int, num_input_channels: int, input_shape,
                 down_channels, up_channels, deep_supervision,
                 voxel_decoder: bool, p_dropout: float = None,
                 skips_only: bool = False):
        assert len(up_channels) == len(down_channels) - 1, \
            "Encoder should have one more step than decoder."
        super().__init__()
        self.num_classes = num_classes
        self.skips_only = skips_only

        self.encoder = ResidualUNetEncoder(num_input_channels, down_channels, p_dropout)
        if voxel_decoder:
            self.decoder = ResidualUNetDecoder(self.encoder, up_channels,
                                               num_classes, deep_supervision,
                                               p_dropout, input_shape,
                                               skips_only)
        else:
            self.decoder = None

    def forward(self, x):
        encoder_skips = self.encoder(x)
        if self.decoder is not None:
            if self.skips_only:
                decoder_skips = self.decoder(encoder_skips)
                return encoder_skips, decoder_skips
            else:
                decoder_skips, seg_out = self.decoder(encoder_skips)
        else:
            decoder_skips, seg_out = [], []

        return encoder_skips, decoder_skips, seg_out
