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
    _C.MODEL.EFFICIENTNET.NAME = "efficientnet-b0"
    _C.MODEL.EFFICIENTNET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]


def add_ghostnet_config(cfg):
    # ---------------------------------------------------------------------------- #
    # EfficientNet options
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.MODEL.GHOSTNET = CN()
    _C.MODEL.GHOSTNET.FEATURE_INDICES = [3, 5, 11, 16]
    _C.MODEL.GHOSTNET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]

def add_hardnet_config(cfg):
    # ---------------------------------------------------------------------------- #
    # EfficientNet options
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.MODEL.HARDNET = CN()
    _C.MODEL.HARDNET.DEPTH_WISE = False
    _C.MODEL.HARDNET.ARCH = 68
    _C.MODEL.HARDNET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]

def add_mobilenet_config(cfg):
    # ---------------------------------------------------------------------------- #
    # EfficientNet options
    # These options apply to both
    # ---------------------------------------------------------------------------- #
    _C = cfg
    _C.MODEL.MOBILENET = CN()
    _C.MODEL.MOBILENET.FEATURE_INDICES = [3, 6, 13, 17]
    _C.MODEL.MOBILENET.OUT_FEATURES = ["stride4", "stride8", "stride16", "stride32"]
    _C.MODEL.MOBILENET.QUANTIZE = False

def add_backbone_config(cfg):
    cfg.MODEL.FPN.REPEAT = 2
    add_vovnet_config(cfg)
    add_efficientnet_config(cfg)
    add_ghostnet_config(cfg)
    add_mobilenet_config(cfg)
    add_hardnet_config(cfg)

__all__ = ['add_backbone_config']
