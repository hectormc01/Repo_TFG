import os
import math
import argparse
import numpy as np
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[1,2,3,5,10,30], help='')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results_attr.txt'), 'w')

    for shot in args.shot_list:

        file_paths = []
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            if fname.split('_')[0] != '{}shot'.format(shot):
                continue
            _dir = os.path.join(args.res_dir, fname)
            if not os.path.isdir(_dir):
                continue
            file_paths.append(os.path.join(_dir, 'log.txt'))

        header = ["AP att", "AP att cor", "AP att pat", "AP att mat", "AP att ref"]
        results = []
        for fid, fpath in enumerate(sorted(file_paths)):
            lineinfos = open(fpath).readlines()
            res_info = lineinfos[-5]
            results.append([fid] + [
                float( res_info[res_info.find("'obj-attr-AP': ")+len("'obj-attr-AP': ") : res_info.find("'obj-attr-AP': ")+len("'obj-attr-AP': ")+12] ),
                float( res_info[res_info.find("'color': ")+len("'color': ") : res_info.find("'color': ")+len("'color': ")+12] ),
                float( res_info[res_info.find("'pattern_marking': ")+len("'pattern_marking': ") : res_info.find("'pattern_marking': ")+len("'pattern_marking': ")+12] ),
                float( res_info[res_info.find("'material': ")+len("'material': ") : res_info.find("'material': ")+len("'material': ")+12] ),
                float( res_info[res_info.find("'transparency': ")+len("'transparency': ") : res_info.find("'transparency': ")+len("'transparency': ")+12] )
            ])

        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
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

    print('Reformat attr results -> {}'.format(os.path.join(args.res_dir, 'results_attr.txt')))


if __name__ == '__main__':
    main()
