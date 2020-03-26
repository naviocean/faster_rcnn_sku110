"""
@Filename : shufflenet
@Date : 2020-03-16
@Project: detectron2
@AUTHOR : NaviOcean
"""
import torch
from torch import nn
from torch.nn import BatchNorm2d
from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from torchvision.models.utils import load_state_dict_from_url


model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                FrozenBatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(Backbone):
    def __init__(self, cfg, stages_repeats, stages_out_channels, num_classes=1000,
                 inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.features = nn.ModuleList([nn.Sequential(
            Conv2d(input_channels, output_channels, 3, 2, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )])
        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, stride=2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, stride=1))
            # setattr(self, name, nn.Sequential(*seq))
            self.features.append(nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.features.append(nn.Sequential(
            Conv2d(input_channels, output_channels, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        ))
        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)
    def _freeze_backbone(self, freeze_at):
        for layer_index in range(freeze_at):
            for p in self.features[layer_index].parameters():
                p.requires_grad = False

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

    def forward(self, x):
        return_features_indices = [1, 2, 3, 4]
        res = []
        for i, m in enumerate(self.features):
            x = m(x)
            if i in return_features_indices:
                res.append(x)
        return {'res{}'.format(i + 2): r for i, r in enumerate(res)}


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_x1_0_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a MobileNetV2 instance from config.
    Returns:
        MobileNetV2: a :class:`MobileNetV2` instance.
    """
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    print(cfg.MODEL.RESNETS.OUT_FEATURES)
    #
    out_feature_channels = {"res2": 116, "res3": 232,
                            "res4": 464, "res5": 1024}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    model = ShuffleNetV2(cfg, [4, 8, 4], [24, 116, 232, 464, 1024])
    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides
    return model

@BACKBONE_REGISTRY.register()
def build_shufflenetv2_x0_5_backbone(cfg, input_shape: ShapeSpec):
    """
    Create a MobileNetV2 instance from config.
    Returns:
        MobileNetV2: a :class:`MobileNetV2` instance.
    """
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    print(cfg.MODEL.RESNETS.OUT_FEATURES)
    #
    out_feature_channels = {"res2": 48, "res3": 96,
                            "res4": 192, "res5": 1024}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}

    model = ShuffleNetV2(cfg, [4, 8, 4], [24, 48, 96, 192, 1024])
    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides
    return model

@BACKBONE_REGISTRY.register()
def build_shufflenetv2_x1_0_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_shufflenetv2_x1_0_backbone(cfg, input_shape)
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
def build_shufflenetv2_x0_5_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_shufflenetv2_x0_5_backbone(cfg, input_shape)
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
    from config import add_vovnet_config
    from detectron2.engine import default_setup, DefaultTrainer
    from detectron2.modeling import build_model
    import collections

    def setup():
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        add_vovnet_config(cfg)
        cfg.merge_from_file('projects/sku110/configs/faster_rcnn_ShuffleNetv2_05_FPNLite_1x.yaml')
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.freeze()
        default_setup(cfg, {})
        return cfg


    cfg = setup()
    net = build_model(cfg)
    source_state = load_state_dict_from_url(model_urls['shufflenetv2_x0.5'], progress=True)
    # source_state = torch.load('./shufflenetv2_x1.pth')
    # print(source_state.keys())
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()

    map = {'features.0': 'conv1' , 'features.1': 'stage2', 'features.2': 'stage3', 'features.3': 'stage4', 'features.4': 'conv5'}
    # print(source_state.keys())
    for target_key, target_value in target_state.items():
        if 'backbone.bottom_up' in target_key:
            key = target_key.split('backbone.bottom_up.')
            feature_layer_key = key[1].split('.')
            feature_layer_key = f'{feature_layer_key[0]}.{feature_layer_key[1]}'
            search_key = key[1].replace(feature_layer_key, map[feature_layer_key])
            if search_key in source_state.keys():
                print(search_key)
                new_target_state[target_key] = source_state[search_key]
            else:
                # print(key[1])
                new_target_state[target_key] = target_state[target_key]
                print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
    print('load new state')
    net.load_state_dict(new_target_state, strict=False)
    #
    torch.save(net.state_dict(), 'shufflenetv2_x0.5.pth')
