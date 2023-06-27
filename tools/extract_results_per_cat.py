import os
import math
import argparse
import numpy as np
from tabulate import tabulate

PACO_NOVEL_CATEGORIES = [{'name': 'ball', 'id': 41, 'image_count': 293, 'instance_count': 739, 'synset': 'ball.n.06', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'bicycle', 'id': 94, 'image_count': 1763, 'instance_count': 4344, 'synset': 'bicycle.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'car_(automobile)', 'id': 207, 'image_count': 1820, 'instance_count': 9978, 'synset': 'car.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'clock', 'id': 271, 'image_count': 1760, 'instance_count': 2564, 'synset': 'clock.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'dog', 'id': 378, 'image_count': 1823, 'instance_count': 2542, 'synset': 'dog.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'earphone', 'id': 409, 'image_count': 514, 'instance_count': 732, 'synset': 'earphone.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'fan', 'id': 429, 'image_count': 549, 'instance_count': 708, 'synset': 'fan.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'knife', 'id': 615, 'image_count': 1778, 'instance_count': 3351, 'synset': 'knife.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'ladder', 'id': 621, 'image_count': 607, 'instance_count': 947, 'synset': 'ladder.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'microwave_oven', 'id': 687, 'image_count': 948, 'instance_count': 1054, 'synset': 'microwave.n.02', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'mirror', 'id': 694, 'image_count': 1814, 'instance_count': 3347, 'synset': 'mirror.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'mouse_(computer_equipment)', 'id': 705, 'image_count': 1277, 'instance_count': 1753, 'synset': 'mouse.n.04', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'newspaper', 'id': 719, 'image_count': 426, 'instance_count': 1137, 'synset': 'newspaper.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'scarf', 'id': 921, 'image_count': 715, 'instance_count': 1250, 'synset': 'scarf.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'soap', 'id': 979, 'image_count': 462, 'instance_count': 841, 'synset': 'soap.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'spoon', 'id': 1000, 'image_count': 1056, 'instance_count': 1996, 'synset': 'spoon.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'sweater', 'id': 1042, 'image_count': 912, 'instance_count': 1794, 'synset': 'sweater.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'table', 'id': 1050, 'image_count': 1757, 'instance_count': 2647, 'synset': 'table.n.02', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'towel', 'id': 1108, 'image_count': 599, 'instance_count': 2077, 'synset': 'towel.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}, {'name': 'watch', 'id': 1161, 'image_count': 1836, 'instance_count': 2581, 'synset': 'watch.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[1,2,3,5,10,30], help='')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results_per_cat.txt'), 'w')

    for shot in args.shot_list:

        file_paths = []
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            if fname.split('_')[0] != '{}shot'.format(shot):
                continue
            _dir = os.path.join(args.res_dir, fname)
            if not os.path.isdir(_dir):
                continue
            file_paths.append(os.path.join(_dir, 'log.txt'))

        header = [cat['name'] for cat in PACO_NOVEL_CATEGORIES]
        results = []
        for fid, fpath in enumerate(sorted(file_paths)):
            lineinfos = open(fpath).readlines()
            res_info = lineinfos[-5]
            results.append([fid] + [
                float( res_info[res_info.find("'{}': ".format(cat['name']))+len("'{}': ".format(cat['name'])) : res_info.find("'{}': ".format(cat['name']))+len("'{}': ".format(cat['name']))+12] )
                if res_info[res_info.find("'{}': ".format(cat['name']))+len("'{}': ".format(cat['name'])) : res_info.find("'{}': ".format(cat['name']))+len("'{}': ".format(cat['name']))+3] != '0.0'
                else float('0.0')
                for cat in PACO_NOVEL_CATEGORIES
            ])

        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        cid = [s for s in np.std(results_np, axis=0)]
        results.append(['μ'] + avg[1:])
        results.append(['c'] + cid[1:])

        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header,
            numalign="left",
        )

        wf.write('--> {}-shot\n'.format(shot))
        wf.write('{}\n\n'.format(table))
        wf.flush()
    wf.close()

    print('Reformat detection results per cat -> {}'.format(os.path.join(args.res_dir, 'results_per_cat.txt')))


if __name__ == '__main__':
    main()
