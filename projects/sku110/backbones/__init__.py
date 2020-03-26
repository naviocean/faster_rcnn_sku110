# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
from .config import add_vovnet_config
from .vovnet import build_vovnet_fpn_backbone, build_vovnet_backbone
from .mobilenet import build_mobilenetv2_fpn_backbone, build_mnv2_backbone
from .shufflenet import build_shufflenetv2_x0_5_fpn_backbone, build_shufflenetv2_x1_0_fpn_backbone