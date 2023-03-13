import json
import os
import sys

for filename in sys.argv[1:]:
    with open(filename, 'r') as f:
        data = json.load(f)

        # Remove part categories
        data['categories'] = [cat for cat in data['categories'] if cat['supercategory'] == 'OBJECT']

        # Remove part category annotations
        obj_cat_ids = [cat['id'] for cat in data['categories']]
        data['annotations'] = [anno for anno in data['annotations'] if anno['category_id'] in obj_cat_ids]

    os.remove(filename)
    with open(filename, 'w') as f:
        json.dump(data, f)