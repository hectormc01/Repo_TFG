import os
import torch
import argparse


def surgery_loop(args, surgery):

    save_name = args.tar_name + '_' + ('remove' if args.method == 'remove' else 'surgery') + '.pth'
    save_path = os.path.join(args.save_dir, save_name)
    os.makedirs(args.save_dir, exist_ok=True)

    ckpt = torch.load(args.src_path)
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0

    if args.method == 'remove':
        for param_name in args.param_name:
            del ckpt['model'][param_name + '.weight']
            if param_name+'.bias' in ckpt['model']:
                del ckpt['model'][param_name+'.bias']
    elif args.method == 'randinit':
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name, tar_size) in enumerate(zip(args.param_name, tar_sizes)):
            surgery(param_name, True, tar_size, ckpt)
            surgery(param_name, False, tar_size, ckpt)
    else:
        raise NotImplementedError

    torch.save(ckpt, save_path)
    print('save changed ckpt to {}'.format(save_path))


def main(args):
    """
    Either remove the final layer weights for fine-tuning on novel dataset or
    append randomly initialized weights for the novel classes.
    """
    def surgery(param_name, is_weight, tar_size, ckpt):
        weight_name = param_name + ('.weight' if is_weight else '.bias')
        pretrained_weight = ckpt['model'][weight_name]
        prev_cls = pretrained_weight.size(0)
        if 'cls_score' in param_name:
            prev_cls -= 1
        if is_weight:
            feat_size = pretrained_weight.size(1)
            new_weight = torch.rand((tar_size, feat_size))
            torch.nn.init.normal_(new_weight, 0, 0.01)
        else:
            new_weight = torch.zeros(tar_size)
        if args.dataset == 'paco':
            for idx, c in enumerate(PACO_BASE_CLASSES):
                if 'cls_score' in param_name:
                    new_weight[IDMAP[c]] = pretrained_weight[idx]
                else:
                    new_weight[IDMAP[c]*4:(IDMAP[c]+1)*4] = \
                        pretrained_weight[idx*4:(idx+1)*4]
        elif args.dataset == 'coco':
            for idx, c in enumerate(COCO_BASE_CLASSES):
                # idx = i if args.dataset == 'coco' else c
                if 'cls_score' in param_name:
                    new_weight[IDMAP[c]] = pretrained_weight[idx]
                else:
                    new_weight[IDMAP[c]*4:(IDMAP[c]+1)*4] = \
                        pretrained_weight[idx*4:(idx+1)*4]
        else:
            new_weight[:prev_cls] = pretrained_weight[:prev_cls]
        if 'cls_score' in param_name:
            new_weight[-1] = pretrained_weight[-1]  # bg class
        ckpt['model'][weight_name] = new_weight

    surgery_loop(args, surgery)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='paco', choices=['voc', 'coco', 'paco'])
    parser.add_argument('--src-path', type=str, default='', help='Path to the main checkpoint')
    parser.add_argument('--save-dir', type=str, default='', required=True, help='Save directory')
    parser.add_argument('--method', choices=['remove', 'randinit'], required=True,
                        help='remove = remove the final layer of the base detector. '
                             'randinit = randomly initialize novel weights.')
    parser.add_argument('--param-name', type=str, nargs='+', help='Target parameter names',
                        default=['roi_heads.box_predictor.cls_score', 'roi_heads.box_predictor.bbox_pred',
                        'roi_heads.attr_predictor.attr_pred1',
                        'roi_heads.attr_predictor.attr_pred2',
                        'roi_heads.attr_predictor.attr_pred3',
                        'roi_heads.attr_predictor.attr_pred4'])
    parser.add_argument('--tar-name', type=str, default='model_reset', help='Name of the new ckpt')
    args = parser.parse_args()

    if args.dataset == 'paco':
        PACO_NOVEL_CLASSES = [41, 94, 207, 271, 378, 409, 429, 615, 621, 687, 694, 705, 719, 921, 979, 1000, 1042, 1050, 1108, 1161]
        PACO_BASE_CLASSES = [23, 35, 61, 88, 90, 112, 127, 133, 139, 143, 156, 160, 184, 192, 220, 230, 232, 324, 344, 396, 399, 498, 521, 530, 544, 556, 591, 604, 626, 631, 708, 713, 751, 781, 782, 804, 811, 818, 821, 881, 898, 923, 926, 948, 973, 999, 1018, 1061, 1072, 1077, 1093, 1117, 1139, 1156, 1196]
        PACO_ALL_CLASSES = sorted(PACO_BASE_CLASSES + PACO_NOVEL_CLASSES)
        IDMAP = {v: i for i, v in enumerate(PACO_ALL_CLASSES)}
        TAR_SIZE = 75
    elif args.dataset == 'coco':
        COCO_NOVEL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 44, 62, 63, 64, 67, 72]
        COCO_BASE_CLASSES = [8, 10, 11, 13, 14, 15, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
                        39, 40, 41, 42, 43, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
                        61, 65, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        COCO_ALL_CLASSES = sorted(COCO_BASE_CLASSES + COCO_NOVEL_CLASSES)
        IDMAP = {v: i for i, v in enumerate(COCO_ALL_CLASSES)}
        TAR_SIZE = 80
    elif args.dataset == 'voc':
        TAR_SIZE = 20
    else:
        raise NotImplementedError

    main(args)
