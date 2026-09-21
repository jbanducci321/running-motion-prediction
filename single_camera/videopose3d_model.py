"""Vendored from facebookresearch/VideoPose3D (common/model.py + the
normalize_screen_coordinates() function from common/camera.py), copied
rather than added as a submodule since only the model architecture itself
is needed here - not the rest of that repo's training/data-prep tooling.

Unlike the rest of this project (MIT licensed), this file is a direct copy
of VideoPose3D's code and is therefore covered by VideoPose3D's own
CC-BY-NC-4.0 license (non-commercial use only) - see
https://github.com/facebookresearch/VideoPose3D/blob/main/LICENSE.

Original copyright (c) 2018-present, Facebook, Inc. All rights reserved.
"""
from __future__ import annotations

import numpy as np
import torch.nn as nn


def normalize_screen_coordinates(coords: np.ndarray, w: int, h: int) -> np.ndarray:
    """Maps pixel coordinates so [0, w] -> [-1, 1], preserving aspect ratio."""
    assert coords.shape[-1] == 2
    return coords / w * 2 - [1, h / w]


class TemporalModelBase(nn.Module):
    """Do not instantiate this class."""

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()

        for fw in filter_widths:
            assert fw % 2 != 0, "Only odd filter widths are supported"

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def receptive_field(self) -> int:
        """Total receptive field of this model, in frames."""
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2 * frames

    def total_causal_shift(self):
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)

        return x


class TemporalModel(TemporalModelBase):
    """Reference 3D pose estimation model with temporal convolutions."""

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024, dense=False):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels, filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0]) // 2 if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2 * next_dilation) if causal else 0)

            layers_conv.append(nn.Conv1d(
                channels, channels,
                filter_widths[i] if not dense else (2 * self.pad[-1] + 1),
                dilation=next_dilation if not dense else 1,
                bias=False,
            ))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            pad = self.pad[i + 1]
            shift = self.causal_shift[i + 1]
            res = x[:, :, pad + shift: x.shape[2] - pad + shift]

            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x
