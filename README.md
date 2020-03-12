# This repository is based on [VoVNet-v2](https://github.com/youngwanLEE/vovnet-detectron2)




### Faster R-CNN on SKU-110K dataset

#### Lightweight-VoVNet with _FPNLite_

|Backbone|Param.|lr sched|inference time|AP|APs|APm|APl|download|
|:--------:|:---:|:---:|:--:|--|----|----|---|--------|
|MobileNetV2|3.5M|3x|0.022|33.0|19.0|35.0|43.4|<a href="https://dl.dropbox.com/s/q4iceofvlcu207c/faster_mobilenetv2_FPNLite_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/tz60e7rtnbsrdgd/faster_mobilenetv2_FPNLite_ms_3x_metrics.json">metrics</a>
||
|V2-19|11.2M|3x|0.034|38.9|24.8|41.7|49.3|<a href="https://www.dropbox.com/s/u5pvmhc871ohvgw/fast_V_19_eSE_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/riu7hkgzlmnndhc/fast_V_19_eSE_FPNLite_ms_3x_metrics.json">metrics</a>
|V2-19-**DW**|6.5M|3x|0.027|36.7|22.7|40.0|46.0|<a href="https://www.dropbox.com/s/7h6zn0owumucs48/faster_rcnn_V_19_eSE_dw_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/627hf4h1m485926/faster_rcnn_V_19_eSE_dw_FPNLite_ms_3x_metrics.json">metrics</a>
|V2-19-**Slim**|3.1M|3x|0.023|35.2|21.7|37.3|44.4|<a href="https://www.dropbox.com/s/yao1i32zdylx279/faster_rcnn_V_19_eSE_slim_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/jrgxltneki9hk84/faster_rcnn_V_19_eSE_slim_FPNLite_ms_3x_metrics.json">metrics</a>
|V2-19-**Slim**-**DW**|1.8M|3x|0.022|32.4|19.1|34.6|41.8|<a href="https://www.dropbox.com/s/blpjx3iavrzkygt/faster_rcnn_V_19_eSE_slim_dw_FPNLite_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://www.dropbox.com/s/3og68zhq2ubr7mu/faster_rcnn_V_19_eSE_slim_dw_FPNLite_ms_3x_metrics.json">metrics</a>

* _**DW** and **Slim** denote depthwise separable convolution and a thiner model with half the channel size, respectively._                              


|Backbone|Param.|lr sched|inference time|AP|APs|APm|APl|download|
|:--------:|:---:|:---:|:--:|--|----|----|---|--------|
|V2-19-FPN|37.6M|3x|0.040|38.9|24.9|41.5|48.8|<a href="https://www.dropbox.com/s/1rfvi6vzx45z6y5/faster_V_19_eSE_ms_3x.pth?dl=1">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/dq7406vo22wjxgi/faster_V_19_eSE_ms_3x_metrics.json">metrics</a>
||
|R-50-FPN|51.2M|3x|0.047|40.2|24.2|43.5|52.0|<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl">model</a>&nbsp;\|&nbsp;<a href="https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/metrics.json">metrics</a>
|**V2-39-FPN**|52.6M|3x|0.047|42.7|27.1|45.6|54.0|<a href="https://dl.dropbox.com/s/dkto39ececze6l4/faster_V_39_eSE_ms_3x.pth">model</a>&nbsp;\|&nbsp;<a href="https://dl.dropbox.com/s/dx9qz1dn65ccrwd/faster_V_39_eSE_ms_3x_metrics.json">metrics</a>
||



Using this command with `--num-gpus 1`
```bash
python /path/to/vovnet_sku110/train_net.py --config-file /path/to/vovnet_sku110/configs/<config.yaml> --eval-only --num-gpus 1 MODEL.WEIGHTS <model.pth>
```

## Installation

As this vovnet-detectron2 is implemented as a [extension form](https://github.com/youngwanLEE/detectron2/tree/vovnet/projects/VoVNet) (detectron2/projects) upon detectron2, you just install [detectron2](https://github.com/facebookresearch/detectron2) following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

Prepare for SKU-110K dataset:
- To download dataset, please visit [here](https://github.com/eg4000/SKU110K_CVPR19)
- Extract the file downloaded to `datasets/sku110/images`
- Extract `datasets/sku110/Annotations.zip`, there are 2 folders `Annotations` and `ImageSets`

## Training

To train a model, run
```bash
python /path/to/vovnet_sku110/train_net.py --config-file /path/to/vovnet_sku110/configs/<config.yaml>
```

For example, to launch end-to-end Faster R-CNN training with VoVNetV2-39 backbone on 8 GPUs,
one should execute:
```bash
python /path/to/vovnet_sku110/train_net.py --config-file /path/to/vovnet_sku110/configs/faster_rcnn_V_39_FPN_3x.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
python /path/to/vovnet_sku110/train_net.py --config-file /path/to/vovnet_sku110/configs/faster_rcnn_V_39_FPN_3x.yaml --eval-only MODEL.WEIGHTS <model.pth>
```

## Visualization
To visual the result, run
```bash
python /path/to/vovnet_sku110/demo.py --config-file /path/to/vovnet_sku110/configs/faster_rcnn_V_39_FPN_3x.yaml --input image.jpg --output image.jpg MODEL.WEIGHTS <model.pth>
```


## <a name="CitingVoVNet"></a>Citing VoVNet

If you use VoVNet, please use the following BibTeX entry.

```BibTeX
@inproceedings{lee2019energy,
  title = {An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection},
  author = {Lee, Youngwan and Hwang, Joong-won and Lee, Sangrok and Bae, Yuseok and Park, Jongyoul},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year = {2019}
}

```
