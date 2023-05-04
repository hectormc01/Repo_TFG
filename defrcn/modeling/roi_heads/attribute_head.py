# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

import torch

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Instances
from torch import nn

from torch.nn import functional as F


__all__ = [
    "MAPPED_ATTRS_LIST",
    "select_proposals_with_attributes",
    "attribute_inference",
    "AttributeOutputLayers",
]


MAPPED_ATTRS_LIST = [
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
    ],
    [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
    [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
    [55, 56, 57, 58],
]


def select_proposals_with_attributes(
    proposals: List[Instances],
    bg_label: int,
) -> List[Instances]:
    """
    Given a list of N Instances (for N images), return a list of Instances that
    contain only instances with `gt_classes != -1 && gt_classes != bg_label`.
    In addition, the returned proposals have at-least one non-ignored attribute
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.
        bg_label: label index of background class.
    Returns:
        proposals: only contains proposals with at least one
        non-ignored attribute.
    """
    ret = []
    all_num_valid_attrs = []
    ignored_sum = []
    for _i, proposals_per_image in enumerate(proposals):
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            all_num_valid_attrs.append(0)
            ret.append(proposals_per_image)
            continue

        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        proposals_per_image = proposals_per_image[fg_selection_mask]

        if len(proposals_per_image) == 0:
            all_num_valid_attrs.append(0)
            ret.append(proposals_per_image)
            continue
        # attr_label_tensor is num_proposals X overall attr. label size
        # attr_ignore_tensor is num_proposals X num_attr_types
        assert proposals_per_image.has("gt_classes")
        assert proposals_per_image.has("gt_attr_label_tensor")
        assert proposals_per_image.has("gt_attr_ignore_tensor")

        attr_ignore_tensor = proposals_per_image.gt_attr_ignore_tensor
        # At-least one attribute type is not ignored
        selection = attr_ignore_tensor.sum(dim=1) < attr_ignore_tensor.shape[1]
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_valid_attrs.append(selection_idxs.numel())

        if len(selection_idxs) > 0:
            valid_attr_ids = nonzero_tuple(attr_ignore_tensor[selection_idxs] > 0)[0]
            ignored_sum.append(valid_attr_ids.numel())
        else:
            ignored_sum.append(0.0)
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("attribute/num_valid_attrs", np.mean(all_num_valid_attrs))
    storage.put_scalar("attribute/num_ignored", np.mean(ignored_sum))
    return ret


def attribute_inference(
    pred_attribute_logits: List[torch.Tensor],
    pred_instances: List[Instances],
):
    """
    Run inference and store different attributes.
    """
    num_boxes_per_image = [len(i) for i in pred_instances]
    merged_probs = []
    for pred_attribute_logit in pred_attribute_logits:
        attribute_probs = pred_attribute_logit.softmax(dim=1)
        merged_probs.append(attribute_probs)
    merged_probs = torch.cat(merged_probs, dim=1)
    merged_probs = merged_probs.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(merged_probs, pred_instances):
        instances.attribute_probs = prob


@torch.jit.unused
def attribute_loss(
    pred_attribute_logits: List[torch.Tensor],
    instances: List[Instances],
    mapped_ids: List[torch.Tensor],
    attr_weight: List[torch.Tensor] = None,
):
    """
    Compute the attribute prediction loss per attribute type.

    Args:
        pred_attribute_logits (List[Tensor]): Each entry "i" in list is
            a tensor of shape (B, A_i) where A_i is num-classes for ith
            attribute
        instances (list[Instances]): A list of N Instances, where N is the
            number of images in the batch. These instances are in 1:1
            correspondence with the pred_mask_logits. The ground-truth
            labels (class, box, mask, attribute) associated with each instance
            are stored in fields.
        mapped_ids: Mapping from attribute ids to attribute types for per
            attribute type softmax
        attr_weight: loss weight for attributes head

    Returns:
        attr_loss (Tensor): A scalar tensor combining the losses.
    """

    gt_attr_classes = [[] for _ in pred_attribute_logits]
    gt_ignore_index = [[] for _ in pred_attribute_logits]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        for attr_idx, _ in enumerate(pred_attribute_logits):
            gt_attr_classes[attr_idx].append(
                instances_per_image.gt_attr_label_tensor[:, mapped_ids[attr_idx]]
            )
            gt_ignore_index[attr_idx].append(
                instances_per_image.gt_attr_ignore_tensor[:, attr_idx]
            )

    per_attr_losses = []
    for attr_idx, _ in enumerate(pred_attribute_logits):
        if len(gt_attr_classes[attr_idx]) == 0:
            per_attr_losses.append(pred_attribute_logits[attr_idx].sum() * 0)
        else:
            gt_attr_tensor = torch.cat(gt_attr_classes[attr_idx], dim=0)
            gt_ignore_tensor = torch.cat(gt_ignore_index[attr_idx], dim=0)

            gt_attr_tensor = gt_attr_tensor[gt_ignore_tensor == 0]
            pred_attr_tensor = pred_attribute_logits[attr_idx][gt_ignore_tensor == 0]

            if len(gt_attr_tensor) == 0:
                per_attr_losses.append(pred_attr_tensor.sum() * 0)
            else:
                if attr_weight is not None:
                    per_attr_losses.append(
                        F.cross_entropy(
                            pred_attr_tensor,
                            gt_attr_tensor,
                            weight=torch.tensor(
                                attr_weight[attr_idx],
                                device=pred_attribute_logits[0].device,
                            ),
                        )
                    )
                else:
                    per_attr_losses.append(
                        F.cross_entropy(pred_attr_tensor, gt_attr_tensor)
                    )

    sum_loss = per_attr_losses[0]
    for pal in per_attr_losses[1:]:
        sum_loss += pal
    return sum_loss

class AttributeOutputLayers(nn.Module):
    """
    One linear layer for each attribute type
    """

    # @configurable
    def __init__(
        self,
        input_shape: ShapeSpec,
        mapped_ids_list: List[int],
        attr_weight: List = [],
        loss_weight: float = 1.0,
    ):
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = (
            input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        )

        self.attr_scores = []
        for attr_type_id, mapped_ids in enumerate(mapped_ids_list):
            linlayer = nn.Linear(input_size, len(mapped_ids))
            nn.init.normal_(linlayer.weight, std=0.01)
            nn.init.constant_(linlayer.bias, 0)
            self.add_module("attr_pred{}".format(attr_type_id + 1), linlayer)
            self.attr_scores.append(linlayer)

        self.loss_weight = loss_weight
        self.mapped_ids_list = mapped_ids_list

        if len(attr_weight) > 0:
            self.attr_weight = attr_weight
        else:
            self.attr_weight = None

    # @classmethod
    # def from_config(cls, _cfg, input_shape, mapped_ids_list):
    #     ret = {
    #         "input_shape": input_shape,
    #         "mapped_ids_list": mapped_ids_list,
    #     }
    #     return ret

    def layers(self, x):
        attr_values = []
        for layer in self.attr_scores:
            attr_values.append(layer(x))
        return attr_values

    def forward(self, x, instances: List[Instances]):
        x = self.layers(x)
        if self.training:
            return {
                "loss_attribute": attribute_loss(
                    x, instances, self.mapped_ids_list, self.attr_weight
                )
                * self.loss_weight
            }
        else:
            attribute_inference(x, instances)
            return instances
