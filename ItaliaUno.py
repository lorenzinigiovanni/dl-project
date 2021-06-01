import types
from typing import Callable, Optional, Union
from torch import tensor, Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ItaliaUno(nn.Module):
    def __init__(self, n_person):
        super(ItaliaUno, self).__init__()

        last_channel = 2048

        self.n_person = n_person
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.outputs = {
            'gender': 2,
            'hair_length': 2,
            'sleeve_lenght': 2,
            'length_lower_body_clothing': 2,
            'type_lower_body_clothing': 2,
            'wearing_hat': 2,
            'carrying_backpack': 2,
            'carrying_bag': 2,
            'carrying_handbag': 2,
            'age': 4,
            'color_upper_body_clothing': 9,
            'color_lower_body_clothing': 10
        }

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            padding=3,
            stride=2
        )

        self.bn1 = self._norm_layer(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = self._make_layer(
            Bottleneck,
            64,
            3
        )

        self.layer2 = self._make_layer(
            Bottleneck,
            128,
            4,
            stride=2
        )

        self.layer3 = self._make_layer(
            Bottleneck,
            256,
            6,
            stride=2
        )

        self.layer4 = self._make_layer(
            Bottleneck,
            512,
            3,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc_layer = nn.Sequential(
            nn.Linear(
                in_features=last_channel,
                out_features=128
            )
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=128,
                out_features=self.n_person
            )
        )

        self.layers = {}

        for k, v in self.outputs.items():
            self.layers[k] = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(
                    in_features=last_channel,
                    out_features=v
                )
            ).to('cuda')

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride
                ),
                norm_layer(
                    planes * block.expansion
                ),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # conv layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # flatten
        x = torch.flatten(x, 1)

        y = self.fc_layer(x)

        output = {
            'pre_classifier': y,
            'internal_id': self.classifier(y)
        }

        for k, v in self.layers.items():
            output[k] = v(x)

        return output

    def get_loss(self, net_output, ground_truth):

        loss = 0.0

        for k in self.outputs:
            loss += F.cross_entropy(
                net_output[k],
                ground_truth[k]
            )

        reid_loss = F.cross_entropy(
            net_output['internal_id'],
            ground_truth['internal_id']
        )

        loss += reid_loss

        return loss

    def get_accuracy(self, predicteds, ground_truth):

        accuracy = 0.0

        for k in self.outputs:
            accuracy += predicteds[k].eq(ground_truth[k]).sum().item()

        reid_accuracy = predicteds['internal_id'].eq(
            ground_truth['internal_id']).sum().item()

        accuracy += reid_accuracy

        return accuracy / (len(self.outputs) + 1)

    def get_test_loss(self, net_output, ground_truth):

        loss = 0.0

        for k in self.outputs:
            loss += F.cross_entropy(
                net_output[k],
                ground_truth[k]
            )

        return loss

    def get_test_accuracy(self, predicteds, ground_truth):

        accuracy = 0.0

        for k in self.outputs:
            accuracy += predicteds[k].eq(ground_truth[k]).sum().item()

        return accuracy / len(self.outputs)
