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


class BasicBlock(nn.Module):
    expansion: int = 1

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
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ReteQuattro(nn.Module):
    def __init__(self):
        super(ReteQuattro, self).__init__()

        last_channel = 512

        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.layer0 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                padding=3,
                stride=2
            ),
            self._norm_layer(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.layer1 = self._make_layer(
            BasicBlock,
            64,
            3
        )

        self.layer2 = self._make_layer(
            BasicBlock,
            128,
            4,
            stride=2
        )

        self.layer3 = self._make_layer(
            BasicBlock,
            256,
            6,
            stride=2
        )

        self.layer4 = self._make_layer(
            BasicBlock,
            512,
            3,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.gender = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.hair_length = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.sleeve_lenght = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.length_lower_body_clothing = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.type_lower_body_clothing = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.wearing_hat = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.carrying_backpack = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.carrying_bag = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.carrying_handbag = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=2
            )
        )

        self.age = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=4
            )
        )

        self.color_upper_body_clothing = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=9
            )
        )

        self.color_lower_body_clothing = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_channel,
                out_features=10
            )
        )

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
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # flatten
        x = torch.flatten(x, 1)

        return {
            'gender': self.gender(x),
            'hair_length': self.hair_length(x),
            'sleeve_lenght': self.sleeve_lenght(x),
            'length_lower_body_clothing': self.length_lower_body_clothing(x),
            'type_lower_body_clothing': self.type_lower_body_clothing(x),
            'wearing_hat': self.wearing_hat(x),
            'carrying_backpack': self.carrying_backpack(x),
            'carrying_bag': self.carrying_bag(x),
            'carrying_handbag': self.carrying_handbag(x),
            'age': self.age(x),
            'color_upper_body_clothing': self.color_upper_body_clothing(x),
            'color_lower_body_clothing': self.color_lower_body_clothing(x)
        }

    def get_loss(self, net_output, ground_truth):

        gender_loss = F.cross_entropy(
            net_output['gender'],
            ground_truth['gender']
        )

        hair_length_loss = F.cross_entropy(
            net_output['hair_length'],
            ground_truth['hair_length']
        )

        sleeve_lenght_loss = F.cross_entropy(
            net_output['sleeve_lenght'],
            ground_truth['sleeve_lenght']
        )

        length_lower_body_clothing_loss = F.cross_entropy(
            net_output['length_lower_body_clothing'],
            ground_truth['length_lower_body_clothing']
        )

        type_lower_body_clothing_loss = F.cross_entropy(
            net_output['type_lower_body_clothing'],
            ground_truth['type_lower_body_clothing']
        )

        wearing_hat_loss = F.cross_entropy(
            net_output['wearing_hat'],
            ground_truth['wearing_hat']
        )

        carrying_backpack_loss = F.cross_entropy(
            net_output['carrying_backpack'],
            ground_truth['carrying_backpack']
        )

        carrying_bag_loss = F.cross_entropy(
            net_output['carrying_bag'],
            ground_truth['carrying_bag']
        )

        carrying_handbag_loss = F.cross_entropy(
            net_output['carrying_handbag'],
            ground_truth['carrying_handbag']
        )

        age_loss = F.cross_entropy(
            net_output['age'],
            ground_truth['age']
        )

        color_upper_body_clothing_loss = F.cross_entropy(
            net_output['color_upper_body_clothing'],
            ground_truth['color_upper_body_clothing']
        )

        color_lower_body_clothing_loss = F.cross_entropy(
            net_output['color_lower_body_clothing'],
            ground_truth['color_lower_body_clothing']
        )

        loss = gender_loss + hair_length_loss + sleeve_lenght_loss + \
            length_lower_body_clothing_loss + type_lower_body_clothing_loss + \
            wearing_hat_loss + carrying_backpack_loss + carrying_bag_loss + \
            carrying_handbag_loss + age_loss + color_upper_body_clothing_loss + \
            color_lower_body_clothing_loss

        return loss, {
            'gender': gender_loss,
            'hair_length': hair_length_loss,
            'sleeve_lenght': sleeve_lenght_loss,
            'length_lower_body_clothing': length_lower_body_clothing_loss,
            'type_lower_body_clothing': type_lower_body_clothing_loss,
            'wearing_hat': wearing_hat_loss,
            'carrying_backpack': carrying_backpack_loss,
            'carrying_bag': carrying_bag_loss,
            'carrying_handbag': carrying_handbag_loss,
            'age': age_loss,
            'color_upper_body_clothing': color_upper_body_clothing_loss,
            'color_lower_body_clothing': color_lower_body_clothing_loss
        }

    def get_accuracy(self, predicteds, ground_truth):

        gender_accuracy = predicteds['gender'].eq(
            ground_truth['gender']).sum().item()
        hair_length_accuracy = predicteds['hair_length'].eq(
            ground_truth['hair_length']).sum().item()
        sleeve_lenght_accuracy = predicteds['sleeve_lenght'].eq(
            ground_truth['sleeve_lenght']).sum().item()
        length_lower_body_clothing_accuracy = predicteds['length_lower_body_clothing'].eq(
            ground_truth['length_lower_body_clothing']).sum().item()
        type_lower_body_clothing_accuracy = predicteds['type_lower_body_clothing'].eq(
            ground_truth['type_lower_body_clothing']).sum().item()
        wearing_hat_accuracy = predicteds['wearing_hat'].eq(
            ground_truth['wearing_hat']).sum().item()
        carrying_backpack_accuracy = predicteds['carrying_backpack'].eq(
            ground_truth['carrying_backpack']).sum().item()
        carrying_bag_accuracy = predicteds['carrying_bag'].eq(
            ground_truth['carrying_bag']).sum().item()
        carrying_handbag_accuracy = predicteds['carrying_handbag'].eq(
            ground_truth['carrying_handbag']).sum().item()
        age_accuracy = predicteds['age'].eq(ground_truth['age']).sum().item()
        color_upper_body_clothing_accuracy = predicteds['color_upper_body_clothing'].eq(
            ground_truth['color_upper_body_clothing']).sum().item()
        color_lower_body_clothing_accuracy = predicteds['color_lower_body_clothing'].eq(
            ground_truth['color_lower_body_clothing']).sum().item()

        accuracy = gender_accuracy + hair_length_accuracy + sleeve_lenght_accuracy + \
            length_lower_body_clothing_accuracy + type_lower_body_clothing_accuracy + \
            wearing_hat_accuracy + carrying_backpack_accuracy + carrying_bag_accuracy + \
            carrying_handbag_accuracy + age_accuracy + color_upper_body_clothing_accuracy + \
            color_lower_body_clothing_accuracy

        return accuracy / 12, {
            'gender': gender_accuracy,
            'hair_length': hair_length_accuracy,
            'sleeve_lenght': sleeve_lenght_accuracy,
            'length_lower_body_clothing': length_lower_body_clothing_accuracy,
            'type_lower_body_clothing': type_lower_body_clothing_accuracy,
            'wearing_hat': wearing_hat_accuracy,
            'carrying_backpack': carrying_backpack_accuracy,
            'carrying_bag': carrying_bag_accuracy,
            'carrying_handbag': carrying_handbag_accuracy,
            'age': age_accuracy,
            'color_upper_body_clothing': color_upper_body_clothing_accuracy,
            'color_lower_body_clothing': color_lower_body_clothing_accuracy
        }
