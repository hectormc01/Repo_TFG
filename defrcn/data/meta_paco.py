import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer

#from paco.data.dataset_mapper import ATTR_TYPE_BG_IDXS
ATTR_TYPE_BG_IDXS = [29, 38, 54, 58]


logger = logging.getLogger(__name__)

__all__ = ["register_meta_paco"]


def load_paco_json(json_file, image_root, meta, dataset_name=None, extra_annotation_keys=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        Same as D2 LVIS dataset
    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info(
            "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
        )

    # Instead of getting metadata from the 3 following lines, we get it as an argument (meta)
    # if dataset_name is not None:
    #     meta = get_instances_meta(dataset_name)
    #     MetadataCatalog.get(dataset_name).set(**meta)

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(
        ann_ids
    ), "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))

    logger.info(
        "Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file)
    )

    if extra_annotation_keys:
        logger.info(
            "The following extra annotation keys will be loaded: {} ".format(
                extra_annotation_keys
            )
        )
    else:
        extra_annotation_keys = []

    def get_file_name(img_root, img_dict):
        # Determine the path from the file_name field
        file_name = img_dict["file_name"]
        return os.path.join(img_root, file_name)

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}

        record["file_name"] = get_file_name(image_root, img_dict)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get(
            "not_exhaustive_category_ids", []
        )
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation
            # file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}

            if dataset_name is not None and "thing_dataset_id_to_contiguous_id" in meta:
                obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                    anno["category_id"]
                ]
            else:
                obj["category_id"] = (
                    anno["category_id"] - 1
                )  # Convert 1-indexed to 0-indexed
            segm = anno["segmentation"]  # list[list[float]]

            if len(segm) == 0:
                continue
            assert len(segm) > 0, segm
            obj["segmentation"] = segm
            for extra_ann_key in extra_annotation_keys:
                obj[extra_ann_key] = anno[extra_ann_key]

            if "attribute_ids" in anno:
                obj["attr_labels"] = anno["attribute_ids"]
                obj["attr_ignores"] = [
                    anno["unknown_color"],
                    anno["unknown_pattern_marking"],
                    anno["unknown_material"],
                    anno["unknown_transparency"],
                ]
            else:
                obj["attr_labels"] = ATTR_TYPE_BG_IDXS
                obj["attr_ignores"] = [1, 1, 1, 1]
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_meta_paco(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_paco_json(annofile, imgdir, metadata, dataset_name=name),
    )

    #
    # Futuro
    # split base-novel
    #

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="lvis",
        **metadata,
    )