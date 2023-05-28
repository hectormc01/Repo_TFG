import logging
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer

from defrcn.dataloader.dataset_mapper import ATTR_TYPE_BG_IDXS


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

    is_shots = "shot" in dataset_name  # few-shot
    if is_shots:
        imgid2info = {}
        shot = dataset_name.split('_')[-2].split('shot')[0]
        seed = int(dataset_name.split('_seed')[-1])
        split_dir = os.path.join('datasets', 'pacosplit', 'seed{}'.format(seed))
        for idx, cls in enumerate(meta["thing_classes"]): # for all classes
            json_file = os.path.join(split_dir, "full_box_{}shot_{}_train.json".format(shot, cls))
            json_file = PathManager.get_local_path(json_file)
            timer = Timer()
            lvis_api = LVIS(json_file)
            if timer.seconds() > 1:
                logger.info(
                    "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
                )
            img_ids = sorted(list(lvis_api.imgs.keys()))
            for img_id in img_ids:
                if img_id not in imgid2info:
                    imgid2info[img_id] = [lvis_api.load_imgs([img_id])[0], lvis_api.img_ann_map[img_id]]
                else:
                    for item in lvis_api.imgToAnns[img_id]:
                        imgid2info[img_id][1].append(item)
        imgs, anns = [], []
        for img_id in imgid2info:
            imgs.append(imgid2info[img_id][0])
            anns.append(imgid2info[img_id][1])
    else:
        json_file = PathManager.get_local_path(json_file)
        timer = Timer()
        lvis_api = LVIS(json_file)
        if timer.seconds() > 1:
            logger.info(
                "Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds())
            )
        # sort indices for reproducible results
        img_ids = sorted(lvis_api.imgs.keys())
        imgs = lvis_api.load_imgs(img_ids)
        anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # As we duplicated some train annotations to ensure every cat has at least 30 instances, we are not going to check that annos are unique
    # ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    # assert len(set(ann_ids)) == len(
    #     ann_ids
    # ), "Annotation ids in '{}' are not unique".format(json_file)

    imgs_anns = list(zip(imgs, anns))
    id_map = meta["thing_dataset_id_to_contiguous_id"]

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
            assert anno["image_id"] == image_id

            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS, "category_id": anno["category_id"]}

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

            if obj["category_id"] in id_map:
                obj["category_id"] = id_map[obj["category_id"]]
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_meta_paco(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_paco_json(annofile, imgdir, metadata, dataset_name=name),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="lvis",
        **metadata,
    )