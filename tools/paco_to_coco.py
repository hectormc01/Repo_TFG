import json
import os
import sys

for filename in sys.argv[1:]:
    with open(filename, 'r') as f:
        data = json.load(f)
        for anno in data['annotations']:
            anno['iscrowd'] = 0

    os.remove(filename)
    with open(filename, 'w') as f:
        json.dump(data, f)