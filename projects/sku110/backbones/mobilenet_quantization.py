# taken from https://github.com/tonylins/pytorch-mobilenet-v2/
# Published by Ji Lin, tonylins
# licensed under the  Apache License, Version 2.0, January 2004
import torch
from torch import nn
from detectron2.layers import  ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.quantization.utils import _replace_relu, quantize_model

__all__ = ['QuantizableMobileNetV2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class QuantizableInvertedResidual(InvertedResidual):
    def __init__(self, *args, **kwargs):
        super(QuantizableInvertedResidual, self).__init__(*args, **kwargs)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.conv(x))
        else:
            return self.conv(x)

    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d:
                fuse_modules(self.conv, [str(idx), str(idx + 1)], inplace=True)


class QuantizableMobileNetV2(Backbone):
    def __init__(self, cfg,
                 width_mult=1.0,
                 round_nearest=8,
                 ):
        super(QuantizableMobileNetV2, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        block = QuantizableInvertedResidual

        norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)

        self.return_features_indices = cfg.MODEL.MOBILENET.FEATURE_INDICES
        self.return_features_num_channels = []

        self.features = nn.ModuleList([ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)])
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel

                if len(self.features) - 1 in self.return_features_indices:
                    self.return_features_num_channels.append(output_channel)

        # building last several layers
        self.features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        if len(self.features) - 1 in self.return_features_indices:
            self.return_features_num_channels.append(self.last_channel)
        self._out_feature_strides = {"stride4": 4, "stride8": 8, "stride16": 16, "stride32": 32}
        self._out_feature_channels = {k: c for k, c in
                                      zip(self._out_feature_strides.keys(), self.return_features_num_channels)}

        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)

    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        features = []
        for i, m in enumerate(self.features):
            x = self.quant(x)
            x = m(x)
            x = self.dequant(x)
            if i in self.return_features_indices:
                features.append(x)
        assert len(self._out_feature_strides.keys()) == len(features)
        return dict(zip(self._out_feature_strides.keys(), features))

        # x = self.quant(x)
        # x = self.features(x)
        # x = self.dequant(x)
        # return x

    def forward(self, x):
        return self._forward_impl(x)

    def fuse_model(self):
        for m in self.modules():
            if type(m) == ConvBNReLU:
                fuse_modules(m, ['0', '1', '2'], inplace=True)
            if type(m) == QuantizableInvertedResidual:
                m.fuse_model()


def mobilenet_v2(cfg):
    model = QuantizableMobileNetV2(cfg)
    _replace_relu(model)
    if cfg.MODEL.MOBILENET.QUANTIZE:
        backend = 'qnnpack'
        quantize_model(model, backend)
    return model


@BACKBONE_REGISTRY.register()
def build_mobilenetv2_quantization_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    _out_features = cfg.MODEL.MOBILENET.OUT_FEATURES
    bottom_up = mobilenet_v2(cfg)
    bottom_up._out_features = _out_features
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


if __name__ == "__main__":
    from detectron2.config import get_cfg
    from config import add_backbone_config
    from detectron2.engine import default_setup, DefaultTrainer
    from detectron2.modeling import build_model
    import collections
    from torchsummary import summary


    def setup():
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        add_backbone_config(cfg)
        cfg.merge_from_file('projects/sku110/configs/faster_rcnn_Mv2_Quantization_FPNLite_1x.yaml')
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.freeze()
        default_setup(cfg, {})
        return cfg


    cfg = setup()
    net = build_model(cfg)
    model_url = 'https://download.pytorch.org/models/quantized/mobilenet_v2_qnnpack_37f702c5.pth' if cfg.MODEL.MOBILENET.QUANTIZE else 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
    source_state = load_state_dict_from_url( model_url
        ,
        map_location=lambda storage, loc: storage, progress=True)
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    # for k, c in target_state.items():
    #     print(k)
    for k, c in source_state.items():
        print(k)
    for target_key, target_value in target_state.items():
        if 'backbone.bottom_up' in target_key:
            key = target_key.split('backbone.bottom_up.')
            if key[1] in source_state:
                new_target_state[target_key] = source_state[key[1]]
                print('loaded ', key[1])
            else:
                # new_target_state[target_key] = target_state[target_key]
                print('[WARNING] Not found pre-trained parameters for {}'.format(key[1]))
        else:
            # new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
    print('load new state')
    net.load_state_dict(new_target_state, strict=False)
    # #
    torch.save(new_target_state, "mobienet_v2.pth")
    print(cfg.MODEL.MOBILENET.QUANTIZE)
    # net = QuantizableMobileNetV2(cfg)
    # summary(net, (3, 224, 224))
    # print(net)
