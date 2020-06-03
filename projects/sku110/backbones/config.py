# -*- coding: utf-8 -*-
# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.

from detectron2.config import CfgNode as CN


def add_vovnet_config(cfg):
    """
    Add config for VoVNet.
    """
    _C = cfg

    _C.MODEL.VOVNET = CN()

    _C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
    _C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

    # Options: FrozenBN, GN, "SyncBN", "BN"
    _C.MODEL.VOVNET.NORM = "FrozenBN"

    _C.MODEL.VOVNET.OUT_CHANNELS = 256

    _C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256


def add_efficientnet_config(cfg):
    # ---------------------------------------------------------------------------- #
    # EfficientNet options
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.MODEL.EFFICIENTNET = CN()
    _C.MODEL.EFFICIENTNET.NAME = "efficientnet_b0"
    _C.MODEL.EFFICIENTNET.FEATURE_INDICES = [1, 4, 10, 15]
    _C.MODEL.EFFICIENTNET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]


def add_backbone_config(cfg):
    cfg.MODEL.FPN.REPEAT = 2
    add_vovnet_config(cfg)
    add_efficientnet_config(cfg)

__all__ = ['add_backbone_config']