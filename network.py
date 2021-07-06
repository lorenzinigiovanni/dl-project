import torch
import torch.nn as nn
import torch.nn.functional as F
from bottleneck_layer import Bottleneck, conv1x1


class Network(nn.Module):
    def __init__(self, n_person):
        super(Network, self).__init__()

        last_channel = 2048
        last_last_channel = 512

        self.n_person = n_person
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.attributes = {
            'gender': 2,
            'hair_length': 2,
            'sleeve_length': 2,
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
                out_features=last_last_channel
            ),
            nn.BatchNorm1d(
                last_last_channel
            ),
            nn.ReLU(inplace=True)
        )

        attribute_dim = 0

        self.layers = nn.ModuleDict()

        for k, v in self.attributes.items():
            self.layers[k] = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(
                    in_features=last_last_channel,
                    out_features=v
                )
            )

            attribute_dim += v

        self.idfier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(
                in_features=last_last_channel + attribute_dim,
                out_features=self.n_person
            )
        )

        self.confidence = nn.Sequential(
            nn.Linear(
                in_features=attribute_dim,
                out_features=attribute_dim
            ),
            nn.Sigmoid()
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

        output = {}
        for k, v in self.layers.items():
            output[k] = v(y)

        concatenate = torch.cat([output[k] for k, _ in self.layers.items()], dim=1)
        re_weighted = self.confidence(concatenate) * concatenate

        pre_id = torch.cat((re_weighted, y), dim=1)

        output['pre_id'] = pre_id
        output['internal_id'] = self.idfier(pre_id)

        return output

    # loss for training
    def get_loss(self, net_output, ground_truth):

        loss = 0.0

        for k in self.attributes:
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

    # accuracy for training
    def get_accuracy(self, predicteds, ground_truth):

        accuracy = 0.0

        for k in self.attributes:
            accuracy += predicteds[k].eq(ground_truth[k]).sum().item()

        reid_accuracy = predicteds['internal_id'].eq(ground_truth['internal_id']).sum().item()

        accuracy += reid_accuracy

        return accuracy / (len(self.attributes) + 1)

    # loss for validation split
    def get_test_loss(self, net_output, ground_truth):

        loss = 0.0

        for k in self.attributes:
            loss += F.cross_entropy(
                net_output[k],
                ground_truth[k]
            )

        return loss

    # accuracy for validation split
    def get_test_accuracy(self, predicteds, ground_truth):

        accuracy = 0.0

        for k in self.attributes:
            accuracy += predicteds[k].eq(ground_truth[k]).sum().item()

        return accuracy / len(self.attributes)
