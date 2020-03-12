##############################################################################
#
# Below code is inspired on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/pascal_voc.py
# --------------------------------------------------------
# Detectron2
# Licensed under the Apache 2.0 license.
# --------------------------------------------------------

from fvcore.common.file_io import PathManager
import os
import numpy as np
import xml.etree.ElementTree as ET

from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

__all__ = ["register_sku110_voc"]

CLASS_NAMES = [
    "object",
]


def load_voc_instances(dirname: str, split: str):
    """
    Load licenseplates VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "annotations", "images"
        split (str): one of "train", "test"
    """
    with PathManager.open(os.path.join(dirname, 'ImageSets/Main', split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=np.str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "images", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            instances.append(
                {"category_id": CLASS_NAMES.index(cls), "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)
    return dicts


def register_sku110_voc(name, dirname, split):
    DatasetCatalog.register(name,
                            lambda: load_voc_instances(dirname, split))
    MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES,
                                  dirname=dirname,
                                  split=split)


if __name__ == "__main__":
    import random
    import cv2
    from detectron2.utils.visualizer import Visualizer
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    dataset_name = f"sku110_{args.split}"
    register_sku110_voc(dataset_name, "datasets/sku110", args.split)
    dataset_dicts = DatasetCatalog.get(dataset_name)
    for d in random.sample(dataset_dicts, args.samples):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1],
                                metadata=MetadataCatalog.get(dataset_name),
                                scale=args.scale)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow(dataset_name, vis.get_image()[:, :, ::-1])

        # Exit? Press ESC
        if cv2.waitKey(0) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
