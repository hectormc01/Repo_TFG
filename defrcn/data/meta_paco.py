#import io
#import os
#import contextlib
#import numpy as np
#from pycocotools.coco import COCO
#from detectron2.structures import BoxMode
#from fvcore.common.file_io import PathManager

from .meta_coco import load_coco_json
from detectron2.data import DatasetCatalog, MetadataCatalog

#import logging
#from fvcore.common.timer import Timer
#from paco.data.dataset_mapper import ATTR_TYPE_BG_IDXS
#ATTR_TYPE_BG_IDXS = [29, 38, 54, 58]


#logger = logging.getLogger(__name__)

__all__ = ["register_meta_paco"]


#
# Futuro
# Adaptación load_coco_json con atributos
#


def register_meta_paco(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_coco_json(annofile, imgdir, metadata, name),
    )

    #
    # Futuro
    # split base-novel
    #

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        **metadata,
    )