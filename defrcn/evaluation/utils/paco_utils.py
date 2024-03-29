# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_utils


def calculate_part_mask_area_outside_obj(dt, gt):
    overlap_precision = np.zeros((len(dt), len(gt)))
    for _i, _dt in enumerate(dt):
        for _j, _gt in enumerate(gt):
            overlap_precision[_i, _j] = _compute_mask_area_outside_impl(_dt, _gt)
    return overlap_precision


def calculate_part_box_area_outside_obj(dt, gt):
    overlap_precision = np.zeros((len(dt), len(gt)))
    for _i, _dt in enumerate(dt):
        for _j, _gt in enumerate(gt):
            overlap_precision[_i, _j] = _compute_box_area_outside_impl(_dt, _gt)
    return overlap_precision


def _compute_mask_area_outside_impl(mask_obj, mask_part):
    mask_obj = mask_utils.decode(mask_obj)
    mask_part = mask_utils.decode(mask_part)
    intersection = np.logical_and(mask_obj, mask_part).sum()
    area_part = mask_part.sum()
    return intersection.sum() / (area_part - intersection + 1e-7)


def _compute_box_area_outside_impl(box_obj, box_part):
    x1 = max(box_obj[0], box_part[0])
    y1 = max(box_obj[1], box_part[1])
    x2 = min(box_obj[2] + box_obj[0], box_part[2] + box_part[0])
    y2 = min(box_obj[3] + box_obj[1], box_part[3] + box_part[1])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_part = box_part[2] * box_part[3]
    return intersection / (area_part - intersection + 1e-7)


def get_mean_AP(aps):
    aps = np.array(aps)
    return np.mean(aps[aps > -1])


def get_AP_from_precisions(precisions, idx):
    precision = precisions[:, :, idx, 0]
    precision = precision[precision > -1]
    return np.mean(precision) if precision.size else float("nan")


def compute_per_attr_map(precisions, attr_to_cont_itds_map):
    """
    Generates AP for each attr class by averaging over the AP for a given attr
    class across all te objects/parts that the attr occurs with in gt
    """
    per_attr_map = {}
    for attr_id, cont_ids in attr_to_cont_itds_map.items():
        res_for_attr_ids = []
        for idx in cont_ids:
            ap = get_AP_from_precisions(precisions, idx)
            res_for_attr_ids.append(float(ap * 100))
        per_attr_map[attr_id] = get_mean_AP(res_for_attr_ids)
    return per_attr_map


def compute_per_attr_type_map(per_attr_map, attr_idx_to_attr_type):
    """
    Takes per attr class results and aggregates them to generate per attr type
    results
    """
    attr_type_map = {}
    for attr_type, attr_ids in attr_idx_to_attr_type.items():
        map_for_rel_attrs = []
        for attr_id in attr_ids:
            if attr_id in per_attr_map.keys():
                map_for_rel_attrs.append(per_attr_map[attr_id])
        map_for_rel_attrs = get_mean_AP(map_for_rel_attrs)
        attr_type_map[attr_type] = map_for_rel_attrs
    return attr_type_map


def heirachrichal_APs(lvis_eval):
    results_processed = {}
    joint_obj_attribute_categories = sorted(
        lvis_eval.lvis_gt.dataset["joint_obj_attribute_categories"],
        key=lambda x: x["obj-attr"],
    )
    joint_attr_id_to_cont_id_eval = {
        attr["obj-attr"]: _i for _i, attr in enumerate(joint_obj_attribute_categories)
    }
    # object cats
    obj_cats = lvis_eval.params.cat_ids
    # all 531 object and object-part cats
    all_obj_names = lvis_eval.lvis_gt.obj_names
    # 55 attr classes
    attr_names = lvis_eval.lvis_gt.attr_names
    # map indices in eval results to joint attrs
    cont_id_eval_to_joint_attr_name = {
        _i: f"{all_obj_names[attr['obj']]}:{attr_names[attr['attr']]}"
        for _i, attr in enumerate(joint_obj_attribute_categories)
    }
    # get mapping for indices in eval results for each attr class
    attr_to_cont_itds_map_obj = defaultdict(list)
    # attr_to_cont_itds_map_part = defaultdict(list)
    for cats in lvis_eval.lvis_gt.dataset["joint_obj_attribute_categories"]:
        joint_id = cats["obj-attr"]
        attr_id = cats["attr"]
        obj_id = cats["obj"]
        # don't report results for bg attributes
        if (
            attr_names[attr_id] == "other(pattern_marking)"
            or attr_names[attr_id] == "other(material)"
        ):
            continue
        if obj_id in obj_cats:
            # attr_to_cont_itds_map_obj[attr_id].append(
            #     joint_attr_id_to_cont_id_eval[joint_id]
            # )
            attr_to_cont_itds_map_obj[attr_id].append(
                # Índice contiguo da combinación cat-attr
                lvis_eval.params.joint_attr_ids.index(joint_attr_id_to_cont_id_eval[joint_id])
            )
        # else:
        #     attr_to_cont_itds_map_part[attr_id].append(
        #         joint_attr_id_to_cont_id_eval[joint_id]
        #     )

    attr_type_to_attr_idxs = lvis_eval.lvis_dt.dataset["attr_type_to_attr_idxs"]
    precisions = lvis_eval.eval_joint_attr["precision"]

    # per attr APs for objects and object-parts separately
    per_attr_map_obj = compute_per_attr_map(precisions, attr_to_cont_itds_map_obj)
    # per_attr_map_part = compute_per_attr_map(precisions, attr_to_cont_itds_map_part)

    # per attr type APs for objects and object-parts separately
    per_attr_type_map_obj = compute_per_attr_type_map(
        per_attr_map_obj, attr_type_to_attr_idxs
    )
    # per_attr_type_map_part = compute_per_attr_type_map(
    #     per_attr_map_part, attr_type_to_attr_idxs
    # )

    # report results for each attr type for objects and parts
    results_processed["obj-attrs"] = {}
    for attr_type, _map in per_attr_type_map_obj.items():
        results_processed["obj-attrs"][attr_type] = _map
    # results_processed["obj-part-attrs"] = {}
    # for attr_type, _map in per_attr_type_map_part.items():
    #     results_processed["obj-part-attrs"][attr_type] = _map

    # report overall attr results for objects and parts
    results_processed["obj-attr-AP"] = get_mean_AP(
        list(results_processed["obj-attrs"].values())
    )
    # results_processed["obj-attr-part-AP"] = get_mean_AP(
    #     list(results_processed["obj-part-attrs"].values())
    # )

    # report per attr class results for objects and parts
    per_attr_name_map_obj = {}
    for x, v in per_attr_map_obj.items():
        per_attr_name_map_obj[attr_names[x]] = v
    results_processed["obj-attr-AP-per-class"] = per_attr_name_map_obj
    # per_attr_name_map_part = {}
    # for x, v in per_attr_map_part.items():
    #     per_attr_name_map_part[attr_names[x]] = v
    # results_processed["obj-attr-part-AP-per-class"] = per_attr_name_map_part

    per_pair_map = {}
    for idx in range(precisions.shape[2]):
        precision = precisions[:, :, idx, 0]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_pair_map[cont_id_eval_to_joint_attr_name[idx]] = float(ap * 100)
    results_processed["per-pair-obj-attr-AP"] = per_pair_map
    return results_processed
