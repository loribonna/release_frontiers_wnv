import torch.nn as nn
from graph.gat_model import MAGAT
from backbones.resnet_cond_bn import resnet18 as resnet18_cond_bn
from backbones.resnet import resnet18


class Resnet_baseline(nn.Module):
    def __init__(self, in_channels=12, out_cls=2, pretrained=0, drop_rate=0.2, colorization=0,
                 neighbours_labels=0,
                 use_conditional_bn=False, enable_grad_retain=False):
        super(Resnet_baseline, self).__init__()
        self.enable_grad_retain = enable_grad_retain
        if use_conditional_bn:
            self.model = resnet18_cond_bn(pretrained=pretrained, num_cond_bn_classes=12 if use_conditional_bn else 10)
        else:
            self.model = resnet18(pretrained=pretrained)

        self.use_conditional_bn = use_conditional_bn

        # CHANGE THE FIRST CONV LAYER
        self.in_channels = in_channels
        self.colorization = colorization
        if colorization:
            conv_1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.conv1 = conv_1

        self.use_dropout = 1
        self.dropout = nn.Dropout(drop_rate)

        self.classifier = nn.Linear(512, out_cls)
        # GRAPH
        self.graph = MAGAT(nin=512,
                           nhid=512,
                           nout=1024,
                           alpha=0.2)
        # use or not the labels of the neighbourd
        self.neighbours_labels = neighbours_labels

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, blocks_adj_matrix, neighbours_numbers, extra=None, returnt='out', *kargs, **kwargs):
        # resnet features extraction
        if self.use_conditional_bn:
            features, feature_prepool = self.model.features((x, extra), returnt='both')
        else:
            features, feature_prepool = self.model.features(x, returnt='both')

        if self.enable_grad_retain and feature_prepool.requires_grad:
            feature_prepool.register_hook(self.activations_hook)

        # 2- graph
        features = self.graph(features, blocks_adj_matrix)
        # decide if classify or not the neighbourhood labels
        if not self.neighbours_labels:
            features = features.view(int(features.shape[0] / neighbours_numbers), neighbours_numbers,
                                     features.shape[1])
            features = features[:, 0, :]
        if self.use_dropout:
            features = self.dropout(features)
        # final classifier
        out = self.classifier(features)

        if returnt == 'out':
            return out
        elif returnt == 'both':
            return out, feature_prepool
        else:
            assert False, 'returnt must be either out or both, found {}'.format(returnt)

    # function to set the first channel and load the weights in the correct channels
    def set_weights_conv1(self, in_channels=None):
        if in_channels is not None:
            self.in_channels = in_channels
        conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if self.in_channels == 9:
            # I add the channels without pretrained
            # eventually CLD, with no pretrained weights
            self.model.conv1 = conv1
        elif self.in_channels >= 13:
            # AOT
            conv1.weight.data[:, 0].copy_(self.model.conv1.weight.data[:, 0])
            # B02 B03 B04
            # no pretrained weights
            # B05 B06 B07
            conv1.weight.data[:, 4:7].copy_(self.model.conv1.weight.data[:, 1:4])
            # B8A WVP B11 B12
            conv1.weight.data[:, 7:11].copy_(self.model.conv1.weight.data[:, 5:9])
            # I add the channels without pretrained
            # eventually CLD, with no pretrained weights
            self.model.conv1 = conv1
        elif self.in_channels >= 13 and not self.colorization:
            # B02 <-- Blue
            conv1.weight.data[:, 1].copy_(self.model.conv1.weight.data[:, 2])
            # B03 <-- Green
            conv1.weight.data[:, 2].copy_(self.model.conv1.weight.data[:, 1])
            # B04 <-- Red
            conv1.weight.data[:, 3].copy_(self.model.conv1.weight.data[:, 0])
            self.model.conv1 = conv1
        elif in_channels is not None:
            self.model.conv1 = conv1
        return
