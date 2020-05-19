"""
@Filename : hardnet
@Date : 2020-05-07
@Project: detectron2
@AUTHOR : NaviOcean
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, FrozenBatchNorm2d, ShapeSpec
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from torchvision.models.utils import load_state_dict_from_url

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.data.size(0), -1)


class CombConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=1, stride=1, dropout=0.1, bias=False):
        super().__init__()
        self.add_module('layer1', ConvLayer(in_channels, out_channels, kernel))
        self.add_module('layer2', DWConvLayer(out_channels, out_channels, stride=stride))

    def forward(self, x):
        return super().forward(x)


class DWConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1, bias=False):
        super().__init__()
        out_ch = out_channels

        groups = in_channels
        kernel = 3
        # print(kernel, 'x', kernel, 'x', out_channels, 'x', out_channels, 'DepthWise')

        self.add_module('dwconv', Conv2d(groups, groups, kernel_size=3,
                                         stride=stride, padding=1, groups=groups, bias=bias))
        self.add_module('norm', FrozenBatchNorm2d(groups))

    def forward(self, x):
        return super().forward(x)


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dropout=0.1, bias=False):
        super().__init__()
        out_ch = out_channels
        groups = 1
        # print(kernel, 'x', kernel, 'x', in_channels, 'x', out_channels)
        self.add_module('conv', Conv2d(in_channels, out_ch, kernel_size=kernel,
                                       stride=stride, padding=kernel // 2, groups=groups, bias=bias))
        self.add_module('norm', FrozenBatchNorm2d(out_ch))
        self.add_module('relu', nn.ReLU6(True))

    def forward(self, x):
        return super().forward(x)


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels

    def __init__(self, in_channels, growth_rate, grmul, n_layers, keepBase=False, residual_out=False, dwconv=False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            use_relu = residual_out
            if dwconv:
                layers_.append(CombConvLayer(inch, outch))
            else:
                layers_.append(ConvLayer(inch, outch))

            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        # print("Blk out =",self.out_channels)
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or \
                    (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out


class HarDNet(Backbone):
    def __init__(self, cfg, depth_wise=False, pretrained=True, weight_path=''):
        super().__init__()
        self.depth_wise = depth_wise
        self.cfg = cfg
        first_ch = [32, 64]
        second_kernel = 3
        max_pool = True
        first_ch = [24, 48]
        ch_list = [96, 320, 640, 1024]
        grmul = 1.6
        gr = [16, 20, 64, 160]
        n_layers = [4, 16, 8, 4]
        downSamp = [1, 1, 1, 0]

        if depth_wise:
            second_kernel = 1
            max_pool = False

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(
            ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3,
                      stride=2, bias=False))

        # Second Layer
        self.base.append(ConvLayer(first_ch[0], first_ch[1], kernel=second_kernel))

        # Maxpooling or DWConv3x3 downsampling
        if max_pool:
            self.base.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.base.append(DWConvLayer(first_ch[1], first_ch[1], stride=2))

        # Build all HarDNet blocks
        ch = first_ch[1]
        for i in range(blks):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i], dwconv=depth_wise)
            ch = blk.get_out_ch()
            self.base.append(blk)

            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                if max_pool:
                    self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    self.base.append(DWConvLayer(ch, ch, stride=2))

        self._initialize_weights()
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)

    def forward(self, x):
        self.return_features_indices = [4, 7, 10, 13]
        res = []
        for i, m in enumerate(self.base):
            x = m(x)

            if i in self.return_features_indices:
                res.append(x)
        return {'res{}'.format(i + 2): r for i, r in enumerate(res)}

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
            elif isinstance(m, FrozenBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


@BACKBONE_REGISTRY.register()
def build_hardnet39_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    out_feature_channels = {"res2": 96, "res3": 320,
                            "res4": 640, "res5": 1024}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    bottom_up = HarDNet(cfg, depth_wise=False)
    bottom_up._out_features = out_features
    bottom_up._out_feature_channels = out_feature_channels
    bottom_up._out_feature_strides = out_feature_strides

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
def build_hardnet39ds_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES

    out_feature_channels = {"res2": 96, "res3": 320,
                            "res4": 640, "res5": 1024}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    bottom_up = HarDNet(cfg, depth_wise=True)
    bottom_up._out_features = out_features
    bottom_up._out_feature_channels = out_feature_channels
    bottom_up._out_feature_strides = out_feature_strides

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
    from torchsummary import summary


    def setup():
        """
        Create configs and perform basic setups.
        """
        cfg = get_cfg()
        cfg.MODEL.DEVICE = 'cpu'
        add_vovnet_config(cfg)
        cfg.merge_from_file('projects/sku110/configs/faster_rcnn_HarDNet_39DS_FPNLite_1x.yaml')
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.freeze()
        default_setup(cfg, {})
        return cfg

    cfg = setup()
    net = build_model(cfg)
    source_state = load_state_dict_from_url('https://ping-chao.com/hardnet/hardnet39ds-0e6c6fa9.pth',map_location=lambda storage, loc: storage, progress=True)
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()

    map = {'features.0': 'conv1', 'features.1': 'stage2', 'features.2': 'stage3', 'features.3': 'stage4',
           'features.4': 'conv5'}
    # print(source_state.keys())
    for target_key, target_value in target_state.items():
        if 'backbone.bottom_up' in target_key:
            key = target_key.split('backbone.bottom_up.')
            new_target_state[target_key] = source_state[key[1]]
        else:
            new_target_state[target_key] = target_state[target_key]
            print('[WARNING] Not found pre-trained parameters for {}'.format(target_key))
    print('load new state')
    net.load_state_dict(new_target_state, strict=False)
    #
    torch.save(net.state_dict(), 'hardnet39ds.pth')


    # net = build_model(cfg)
    # print(net)
    # model = build_hardnet39ds_fpn_backbone(cfg)
    # net = HarDNet(cfg, depth_wise=False)
    # summary(net, (3, 224, 224))
    # print(net)
