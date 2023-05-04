import json
import os
import sys

for filename in sys.argv[1:]:
    with open(filename, 'r') as f:
        data = json.load(f)

        # Remove the keys 'part_categories' and 'queries' if they exist
        data.pop('part_categories', None)
        data.pop('queries', None)

        # Remove part categories from 'categories'
        data['categories'] = [cat for cat in data['categories'] if cat['supercategory'] == 'OBJECT']

        # Remove part categories from 'annotations'
        obj_cat_ids = [cat['id'] for cat in data['categories']]
        data['annotations'] = [anno for anno in data['annotations'] if anno['category_id'] in obj_cat_ids]

        # Remove part categories from 'joint_obj_attribute_categories'
        if 'joint_obj_attribute_categories' in data:
            data['joint_obj_attribute_categories'] = [elem for elem in data['joint_obj_attribute_categories'] if elem['obj'] in obj_cat_ids]

    os.remove(filename)
    with open(filename, 'w') as f:
        json.dump(data, f)