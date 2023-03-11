# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Autogen with
# with open("paco_lvis_v1_train.json") as f:
#     PACO_LVIS_IMAGE_COUNT = json.load(f)["categories"][:75]
# PACO_LVIS_IMAGE_COUNT_FINAL = []
# for x in PACO_LVIS_IMAGE_COUNT:
#     PACO_LVIS_IMAGE_COUNT_FINAL.append(
#         {
#             "id": x["id"],
#             "image_count": x["image_count"]
#         }
#     )

# with open("paco_lvis_category_image_count.py", "wt") as f:
#     f.write(f"PACO_LVIS_EGO4D_CATEGORY_IMAGE_COUNT = {PACO_LVIS_IMAGE_COUNT_FINAL}")
# Then paste the contents of that file below

# fmt: off
PACO_LVIS_CATEGORY_IMAGE_COUNT = [{'id': 23, 'image_count': 1791}, {'id': 35, 'image_count': 1775}, {'id': 41, 'image_count': 293}, {'id': 61, 'image_count': 1541}, {'id': 88, 'image_count': 1833}, {'id': 90, 'image_count': 1816}, {'id': 94, 'image_count': 1763}, {'id': 112, 'image_count': 233}, {'id': 127, 'image_count': 1819}, {'id': 133, 'image_count': 1791}, {'id': 139, 'image_count': 1822}, {'id': 143, 'image_count': 1730}, {'id': 156, 'image_count': 91}, {'id': 160, 'image_count': 680}, {'id': 184, 'image_count': 55}, {'id': 192, 'image_count': 425}, {'id': 207, 'image_count': 1820}, {'id': 220, 'image_count': 60}, {'id': 230, 'image_count': 1780}, {'id': 232, 'image_count': 1835}, {'id': 271, 'image_count': 1760}, {'id': 324, 'image_count': 237}, {'id': 344, 'image_count': 1440}, {'id': 378, 'image_count': 1823}, {'id': 396, 'image_count': 18}, {'id': 399, 'image_count': 27}, {'id': 409, 'image_count': 514}, {'id': 429, 'image_count': 549}, {'id': 498, 'image_count': 1830}, {'id': 521, 'image_count': 189}, {'id': 530, 'image_count': 24}, {'id': 544, 'image_count': 1828}, {'id': 556, 'image_count': 1816}, {'id': 591, 'image_count': 402}, {'id': 604, 'image_count': 95}, {'id': 615, 'image_count': 1778}, {'id': 621, 'image_count': 607}, {'id': 626, 'image_count': 1715}, {'id': 631, 'image_count': 1752}, {'id': 687, 'image_count': 948}, {'id': 694, 'image_count': 1814}, {'id': 705, 'image_count': 1277}, {'id': 708, 'image_count': 770}, {'id': 713, 'image_count': 1715}, {'id': 719, 'image_count': 426}, {'id': 751, 'image_count': 216}, {'id': 781, 'image_count': 322}, {'id': 782, 'image_count': 147}, {'id': 804, 'image_count': 1818}, {'id': 811, 'image_count': 1336}, {'id': 818, 'image_count': 1842}, {'id': 821, 'image_count': 28}, {'id': 881, 'image_count': 1039}, {'id': 898, 'image_count': 1392}, {'id': 921, 'image_count': 715}, {'id': 923, 'image_count': 676}, {'id': 926, 'image_count': 48}, {'id': 948, 'image_count': 1809}, {'id': 973, 'image_count': 56}, {'id': 979, 'image_count': 462}, {'id': 999, 'image_count': 83}, {'id': 1000, 'image_count': 1056}, {'id': 1018, 'image_count': 282}, {'id': 1042, 'image_count': 912}, {'id': 1050, 'image_count': 1757}, {'id': 1061, 'image_count': 129}, {'id': 1072, 'image_count': 740}, {'id': 1077, 'image_count': 1814}, {'id': 1093, 'image_count': 306}, {'id': 1108, 'image_count': 599}, {'id': 1117, 'image_count': 889}, {'id': 1139, 'image_count': 1772}, {'id': 1156, 'image_count': 109}, {'id': 1161, 'image_count': 1836}, {'id': 1196, 'image_count': 31}]

# fmt: on
