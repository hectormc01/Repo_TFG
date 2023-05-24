import json
import os
import sys
import random

for filename in sys.argv[1:]:
    with open(filename, 'r') as f:
        data = json.load(f)
        random.seed(123)

        print("Número total de anotacións antes:", len(data['annotations']))

        # Get number of annotations per category
        num_anno_per_cat = {}
        for cat in data['categories']:
            num_anno_per_cat[cat['id']] = sum(1 for anno in data['annotations'] if anno['category_id'] == cat['id'])

        # Duplicate random annotations for categories with less than 30 annotations until reaching 30 instances
        for cat_id in num_anno_per_cat.keys():
            if num_anno_per_cat[cat_id] < 30:
                num_anno_remaining = 30 - num_anno_per_cat[cat_id]
                annotations_original = [anno for anno in data['annotations'] if anno['category_id'] == cat_id]

                for i in range(num_anno_remaining):
                    random_anno = random.choice(annotations_original)
                    annotations_original.remove(random_anno)
                    data['annotations'].append(random_anno)
                
                print(f"Duplicáronse {num_anno_remaining} anotacións da categoría {cat_id} ({[cat['name'] for cat in data['categories'] if cat['id'] == cat_id][0]})")

        print("Número total de anotacións despois:", len(data['annotations']))

    os.remove(filename)
    with open(filename, 'w') as f:
        json.dump(data, f)