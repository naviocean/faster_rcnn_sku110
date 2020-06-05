# taken from https://github.com/tonylins/pytorch-mobilenet-v2/
# Published by Ji Lin, tonylins
# licensed under the  Apache License, Version 2.0, January 2004
import torch
from torch import nn
from torch.nn import BatchNorm2d
from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        Conv2d(inp, oup, 3, stride, 1, bias=False),
        FrozenBatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(context, probs):
        binarized = (probs == torch.max(probs, dim=1, keepdim=True)[0]).float()
        context.save_for_backward(binarized)
        return binarized

    @staticmethod
    def backward(context, gradient_output):
        binarized, = context.saved_tensors
        gradient_output[binarized == 0] = 0
        return gradient_output


class Flgc2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=8, bias=True):
        super().__init__()
        self.in_channels_in_group_assignment_map = nn.Parameter(torch.Tensor(in_channels, groups))
        nn.init.normal_(self.in_channels_in_group_assignment_map)
        self.out_channels_in_group_assignment_map = nn.Parameter(torch.Tensor(out_channels, groups))
        nn.init.normal_(self.out_channels_in_group_assignment_map)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, 1, bias)
        self.binarize = Binarize.apply

    def forward(self, x):
        map = torch.mm(self.binarize(torch.softmax(self.out_channels_in_group_assignment_map, dim=1)),
                       torch.t(self.binarize(torch.softmax(self.in_channels_in_group_assignment_map, dim=1))))
        return nn.functional.conv2d(x, self.conv.weight * map[:, :, None, None], self.conv.bias,
                                    self.conv.stride, self.conv.padding, self.conv.dilation)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, groups_in_1x1, use_flgc=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        # self.use_res_connect = self.stride == 1 and inp == oup
        self.shortcut = nn.Sequential()
        if self.stride == 1 and inp != oup:
            self.shortcut = nn.Sequential(
                Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),  # TODO: add group conv here
                FrozenBatchNorm2d(oup),
            )

        pointwise_conv = nn.Conv2d
        if use_flgc:
            pointwise_conv = Flgc2d

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                FrozenBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                # Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                pointwise_conv(hidden_dim, oup, 1, 1, 0, bias=False, groups=groups_in_1x1),
                FrozenBatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                #
                FrozenBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                FrozenBatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                # Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                pointwise_conv(hidden_dim, oup, 1, 1, 0, bias=False, groups=groups_in_1x1),
                FrozenBatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x) if self.stride == 1 else self.conv(x)


class MobileNetV2(Backbone):
    """
    Should freeze bn
    """

    def __init__(self, cfg, input_size=224, multi=1., groups_in_1x1=1, use_flgc=True):
        print('==================> multi:', multi)
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0

        input_channel = int(input_channel * multi)
        self.return_features_indices = cfg.MODEL.MOBILENET.FEATURE_INDICES
        self.return_features_num_channels = []
        self.features = nn.ModuleList([conv_bn(3, input_channel, 2)])
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * multi)
            for i in range(n):
                if i == 0:
                    self.features.append(
                        block(input_channel, output_channel, s, expand_ratio=t, groups_in_1x1=groups_in_1x1,
                              use_flgc=use_flgc))
                else:
                    self.features.append(
                        block(input_channel, output_channel, 1, expand_ratio=t, groups_in_1x1=groups_in_1x1,
                              use_flgc=use_flgc))
                input_channel = output_channel
                if len(self.features) - 1 in self.return_features_indices:
                    self.return_features_num_channels.append(output_channel)

        self._out_feature_strides = {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32}
        self._out_feature_channels = {k: c for k, c in
                                      zip(self._out_feature_strides.keys(), self.return_features_num_channels)}
        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)


    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def forward(self, x):
        features = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in self.return_features_indices:
                features.append(x)
        assert len(self._out_feature_strides.keys()) == len(features)
        return dict(zip(self._out_feature_strides.keys(), features))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


@BACKBONE_REGISTRY.register()
def build_mnv2_flgc_backbone(cfg, input_shape):

    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    model = MobileNetV2(cfg, groups_in_1x1=8)
    model._out_features = out_features
    return model



@BACKBONE_REGISTRY.register()
def build_mobilenetv2_flgc_05_fpn_backbone(cfg, input_shape: ShapeSpec):
    out_features = cfg.MODEL.MOBILENET.OUT_FEATURES
    bottom_up = MobileNetV2(cfg=cfg, multi=0.5)
    bottom_up._out_features = out_features
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_mobilenetv2_flgc_fpn_backbone(cfg, input_shape: ShapeSpec):
    out_features = cfg.MODEL.MOBILENET.OUT_FEATURES
    bottom_up = MobileNetV2(cfg, groups_in_1x1=8)
    bottom_up._out_features = out_features
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
