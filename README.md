# This repository is based on [VoVNet-v2](https://github.com/youngwanLEE/vovnet-detectron2)



### Faster R-CNN on SKU-110K dataset

### Note

We measure the inference time of all models with batch size 1 on the same RTX2080Ti GPU machine.

- pytorch1.4.0
- CUDA 10.2
- cuDNN 7.3

#### Lightweight with _FPNLite_

|Backbone|Param.|lr sched|inference time|AP|AP75|AP50|download|
|:--------:|:---:|:---:|:--:|--|----|----|--------|
|MobileNetV2-0.5-**64**|N/A|1x|0.033|43.31|44.66|78.08|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|MobileNetV2-0.5|N/A|1x|0.037|42.93|44.27|77.31|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|MobileNetV2|3.5M|3x|0.031|52.11|58.72|85.98|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|MobileNetV2|3.5M|1x|0.031|51.20|56.93|85.71|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|MobileNetV2-FLGC|N/A|1x|0.030|50.59|56.05|85.21|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|ShuffleNetV2-0.5|N/A|1x|0.039|48.24|52.95|82.10|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|ShuffleNetV2|N/A|1x|0.028|52.60|59.55|86.19|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
||
|V2-19|11.2M|1x|0.034|41.46|44.97|71.32|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|V2-19-**DW**|6.5M|1x|N/A|N/A|N/A|N/A|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|V2-19-**Slim**|3.1M|1x|0.027|47.68|51.47|82.36|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|V2-19-**Slim**-**DW**|1.8M|3x|N/A|N/A|N/A|N/A|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>

* **64** FPN.OUT_CHANNELS = 64
* _**DW** and **Slim** denote depthwise separable convolution and a thiner model with half the channel size, respectively._                              
#### _FPN_

|Backbone|Param.|lr sched|inference time|AP|AP75|AP50|download|
|:--------:|:---:|:---:|:--:|--|----|----|--------|
|V2-19-FPN|37.6M|3x|N/A|N/A|N/A|N/A|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
||
|R-50-FPN|51.2M|3x|N/A|N/A|N/A|N/A|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>
|**V2-39-FPN**|52.6M|3x|0.071|51.47|57.5|85.5|<a href="">model</a>&nbsp;\|&nbsp;<a href="">metrics</a>



Using this command with `--num-gpus 1`
```bash
python /path/to/sku110/train_net.py --config-file /path/to/sku110/configs/<config.yaml> --eval-only --num-gpus 1 MODEL.WEIGHTS <model.pth>
```

## Installation

As this repository is implemented as a [extension form](https://github.com/youngwanLEE/detectron2/tree/vovnet/projects/VoVNet) (detectron2/projects) upon detectron2, you just install [detectron2](https://github.com/facebookresearch/detectron2) following [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

Prepare for SKU-110K dataset:
- To download dataset, please visit [here](https://github.com/eg4000/SKU110K_CVPR19)
- Extract the file downloaded to `datasets/sku110/images`
- Extract `datasets/sku110/Annotations.zip`, there are 2 folders `Annotations` and `ImageSets`

## Training

To train a model, run
```bash
python /path/to/sku110/train_net.py --config-file /path/to/sku110/configs/<config.yaml>
```

For example, to launch end-to-end Faster R-CNN training with VoVNetV2-39 backbone on 8 GPUs,
one should execute:
```bash
python /path/to/sku110/train_net.py --config-file /path/to/sku110/configs/faster_rcnn_V_39_FPN_3x.yaml --num-gpus 8
```

## Evaluation

Model evaluation can be done similarly:
```bash
python /path/to/sku110/train_net.py --config-file /path/to/sku110/configs/faster_rcnn_V_39_FPN_3x.yaml --eval-only MODEL.WEIGHTS <model.pth>
```

## Visualization
To visual the result, run
```bash
python /path/to/sku110/demo.py --config-file /path/to/sku110/configs/faster_rcnn_V_39_FPN_3x.yaml --input image.jpg --output image.jpg MODEL.WEIGHTS <model.pth>
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
