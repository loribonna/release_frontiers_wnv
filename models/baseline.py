import torch
import torch.nn as nn
from backbones.resnet import resnet18
from backbones.resnet_cond_bn import resnet18 as resnet18_cond_bn
import torch.nn.functional as F


class AttentionModule(nn.Module):

    def __init__(self, fin, fhidden):  # standard a fin = 512
        super(AttentionModule, self).__init__()

        self.U = nn.Sequential(
            nn.Linear(in_features=fin, out_features=fhidden, bias=False),
            nn.BatchNorm1d(num_features=fhidden),
            nn.Sigmoid()
        )

        self.V = nn.Sequential(
            nn.Linear(in_features=fin, out_features=fhidden, bias=False),
            nn.BatchNorm1d(num_features=fhidden),
            nn.Tanh()
        )

        self.w = nn.Linear(in_features=fhidden, out_features=1, bias=False)

        self.attention_no_gating = nn.Sequential(
            nn.Linear(fin, fhidden),
            nn.Tanh(),
            nn.Linear(fhidden, 1)
        )

    def forward(self, x, gating=1):
        if gating == 1:
            w = self.w(self.V(x) * self.U(x))
        else:
            w = self.attention_no_gating(x)
        return w


class Resnet_baseline(nn.Module):
    def __init__(self, in_channels=12, out_cls=2, pretrained=0, drop_rate=0.2, num_multi_images=1, colorization=0, n_bands=9,
                 use_conditional_bn=False, enable_grad_retain=False):
        super(Resnet_baseline, self).__init__()
        self.enable_grad_retain = enable_grad_retain
        self.use_conditional_bn = use_conditional_bn
        if use_conditional_bn:
            self.model = resnet18_cond_bn(pretrained=pretrained, num_cond_bn_classes=12 if use_conditional_bn else 10)
        else:
            self.model = resnet18(pretrained=pretrained == 1)
        # CHANGE THE FIRST CONV LAYER
        self.in_channels = in_channels
        self.colorization = colorization
        if colorization:
            conv_1 = nn.Conv2d(n_bands, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.model.conv1 = conv_1

        # FEATURE EXTRACTOR
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])

        self.dropout = nn.Dropout(drop_rate)

        self.classifier = nn.Linear(in_features=512, out_features=out_cls)
        # SIGMOID FOR THE MULTILABEL SCENARIO
        self.sigmoid = nn.Sigmoid()
        # MULTI-IMAGES?
        self.num_multi_images = num_multi_images
        # TEMPORAL AGGREGATION
        self.attention_module = AttentionModule(512, 256)

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x, extra=None, returnt='out', *kargs, **kwargs):
        if self.num_multi_images > 1:
            B, N, c, h, w = x.shape
            x = x.view(B * N, c, h, w)

            if self.use_conditional_bn:
                extra = extra.unsqueeze(1).expand(B, N) if len(extra.shape) == 1 else extra
                features, feature_prepool = self.model.features((x, extra), returnt='both')
            else:
                features, feature_prepool = self.model.features(x, returnt='both')

            if self.enable_grad_retain and feature_prepool.requires_grad:
                feature_prepool.register_hook(self.activations_hook)

            features_flatten = features.reshape(features.shape[0], -1)
            features_split = features.reshape(B, N, -1)
            A = self.attention_module(features_flatten).view(B, N)
            A = F.softmax(A, dim=1).view(B, N, 1)
            M = torch.sum(features_split * A, dim=1)
            out = self.classifier(M)
        elif self.num_multi_images == 1:
            if self.use_conditional_bn:
                features, feature_prepool = self.model.features((x, extra.squeeze(-1)), returnt='both')
            else:
                features, feature_prepool = self.model.features(x, returnt='both')
            features = features.view(x.shape[0], -1)
            features = self.dropout(features)
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
        self.model.conv1 = conv1

        self.feature_extractor[0] = conv1
