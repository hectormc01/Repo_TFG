import json
import math
import os
import argparse
from defrcn.evaluation.paco_eval_api import PACO, PACOResults, PACOEval
from defrcn.evaluation.utils.paco_utils import get_AP_from_precisions, get_mean_AP, heirachrichal_APs
from detectron2.utils.logger import create_small_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_dataset_dir', type=str, help='path to the split dataset',
        default='datasets/pacosplit')
        # default='/data/datasets/pacosplit')
    parser.add_argument('--inference_dir', type=str, help='path to the predictions',
        default='/results/proba_defrcn_fewshot_paco_si_attr_freeze_attr/defrcn_fsod_r101_novel/fsrw-like')
        # default='/data/models/proba_defrcn_fewshot_paco_si_attr_freeze_attr/defrcn_fsod_r101_novel/fsrw-like')
    parser.add_argument('--output_dir', type=str, help='path to the output dir',
        default='/proba_postprocess/defrcn_fsod_r101_novel/fsrw-like')
        # default='/home/hector.martinez.casares/proba_postprocess/defrcn_fsod_r101_novel/fsrw-like')
    parser.add_argument('--test_dataset_dir', type=str, help='path to the test dataset',
        default='datasets/paco/annotations/paco_lvis_v1_test.json')
        #default='/data/datasets/paco/annotations/paco_lvis_v1_test.json')
    parser.add_argument('--factors', type=str, help='influence factors to test',
        default=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    args = parser.parse_args()
    return args


PACO_NOVEL_CATEGORIES = [
    {'name': 'ball', 'id': 41, 'image_count': 293, 'instance_count': 739, 'synset': 'ball.n.06', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'bicycle', 'id': 94, 'image_count': 1763, 'instance_count': 4344, 'synset': 'bicycle.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'car_(automobile)', 'id': 207, 'image_count': 1820, 'instance_count': 9978, 'synset': 'car.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'clock', 'id': 271, 'image_count': 1760, 'instance_count': 2564, 'synset': 'clock.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'dog', 'id': 378, 'image_count': 1823, 'instance_count': 2542, 'synset': 'dog.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'earphone', 'id': 409, 'image_count': 514, 'instance_count': 732, 'synset': 'earphone.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'fan', 'id': 429, 'image_count': 549, 'instance_count': 708, 'synset': 'fan.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'knife', 'id': 615, 'image_count': 1778, 'instance_count': 3351, 'synset': 'knife.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'ladder', 'id': 621, 'image_count': 607, 'instance_count': 947, 'synset': 'ladder.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'microwave_oven', 'id': 687, 'image_count': 948, 'instance_count': 1054, 'synset': 'microwave.n.02', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'mirror', 'id': 694, 'image_count': 1814, 'instance_count': 3347, 'synset': 'mirror.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'mouse_(computer_equipment)', 'id': 705, 'image_count': 1277, 'instance_count': 1753, 'synset': 'mouse.n.04', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'newspaper', 'id': 719, 'image_count': 426, 'instance_count': 1137, 'synset': 'newspaper.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'scarf', 'id': 921, 'image_count': 715, 'instance_count': 1250, 'synset': 'scarf.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'soap', 'id': 979, 'image_count': 462, 'instance_count': 841, 'synset': 'soap.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'spoon', 'id': 1000, 'image_count': 1056, 'instance_count': 1996, 'synset': 'spoon.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'sweater', 'id': 1042, 'image_count': 912, 'instance_count': 1794, 'synset': 'sweater.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'table', 'id': 1050, 'image_count': 1757, 'instance_count': 2647, 'synset': 'table.n.02', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'towel', 'id': 1108, 'image_count': 599, 'instance_count': 2077, 'synset': 'towel.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'},
    {'name': 'watch', 'id': 1161, 'image_count': 1836, 'instance_count': 2581, 'synset': 'watch.n.01', 'frequency': 'f', 'supercategory': 'OBJECT'}
]


def cosine_similarity(lista1, lista2):
    # Produto escalar
    dot_product = sum([x * y for x, y in zip(lista1, lista2)])

    # Normas
    norm1 = math.sqrt(sum([x**2 for x in lista1]))
    norm2 = math.sqrt(sum([x**2 for x in lista2]))

    # Similaridade coseno
    cos_sim = dot_product / (norm1 * norm2)

    return cos_sim


def postprocess():
    args = parse_args()

    # Gardamos os atributos das anotacións do soporte que teñen todos os atributos para cada clase novel e cada shot
    # Só da seed 0 porque as probas FSOD realízanse con varias repeticións sobre a mesma seed
    sample = dict()
    for seed in [0]: # for seed in range(10):
        for shot in [1,2,3,5,10,30]:
            for cat in PACO_NOVEL_CATEGORIES:
                data_path = os.path.join(args.split_dataset_dir, 'seed{}/full_box_{}shot_{}_train.json'.format(seed, shot, cat['name']))
                data = json.load(open(data_path))
                # keys: ['images', 'annotations', 'categories', 'attributes', 'attr_type_to_attr_idxs']

                sample['seed{}_{}shot_cat{}'.format(seed, shot, cat['id'])] = []
                for anno in data['annotations']:
                    # keys: ['bbox', 'category_id', 'image_id', 'id', 'segmentation', 'area', 'attribute_ids', 'dom_color_ids', 'unknown_material', 'unknown_transparency', 'obj_ann_id', 'unknown_color', 'unknown_pattern_marking']
                    if((not anno['unknown_material']) and (not anno['unknown_transparency']) and (not anno['unknown_color']) and (not anno['unknown_pattern_marking'])):
                        transformada_col_1 = [1 if i in anno['attribute_ids'] else 0 for i in range(30)]
                        transformada_pat_1 = [1 if i in anno['attribute_ids'] else 0 for i in range(30,41)]
                        transformada_mat_1 = [1 if i in anno['attribute_ids'] else 0 for i in range(41,55)]
                        transformada_ref_1 = [1 if i in anno['attribute_ids'] else 0 for i in range(55,59)]
                        
                        num_col = sum(transformada_col_1)
                        num_pat = sum(transformada_pat_1)
                        num_mat = sum(transformada_mat_1)
                        num_ref = sum(transformada_ref_1)

                        transformada_col_2 = [i/num_col for i in transformada_col_1]
                        transformada_pat_2 = [i/num_pat for i in transformada_pat_1]
                        transformada_mat_2 = [i/num_mat for i in transformada_mat_1]
                        transformada_ref_2 = [i/num_ref for i in transformada_ref_1]

                        transformada = transformada_col_2 + transformada_pat_2 + transformada_mat_2 + transformada_ref_2
                        
                        sample['seed{}_{}shot_cat{}'.format(seed, shot, cat['id'])].append(transformada)

    # Postprocesamos os resultados FSOD
    for repeat_id in [0]: ## for repeat_id in range(10):
        for shot in [1,2,3,5,10,30]:
            for seed in [0]:
                inference_path = os.path.join(args.inference_dir, '{}shot_seed{}_repeat{}/inference/lvis_instances_results.json'.format(shot, seed, repeat_id))
                preds = json.load(open(inference_path))
                
                post_preds = {}
                for factor in args.factors:
                    post_preds[str(factor)] = []
                
                for pred in preds:
                    # keys: ['image_id', 'category_id', 'bbox', 'score', 'attribute_probs']
                    attr_pred = pred['attribute_probs']

                    # attr_color_pred = attr_pred[0:30]           # 0-29 ambos incluidos
                    # attr_pattern_pred = attr_pred[30:41]        # 30-40 ambos incluidos
                    # attr_material_pred = attr_pred[41:55]       # 41-54 ambos incluídos
                    # attr_transparency_pred = attr_pred[55:59]   # 55-58 ambos incluídos

                    # Calcular a similitude entre os atributos preditos para unha categoría e os atributos
                    # anotados para cada un dos exemplos desa clase do shot a través da cosine similarity
                    max_sim = 0
                    for obj_attr in sample['seed{}_{}shot_cat{}'.format(seed, shot, pred['category_id'])]:
                        cos_sim = cosine_similarity(attr_pred, obj_attr)
                        if(cos_sim > max_sim):
                            max_sim = cos_sim
                    
                    for factor in args.factors:
                        # Ponderar a confianza da predición en función da máxima similaridade atopada
                        post_pred = dict(pred)
                        post_score = pred['score'] + factor * max_sim * pred['score']
                        post_pred['score'] = post_score
                        post_preds[str(factor)].append(post_pred)

                # Gardar as post_preds (de cada factor) nun json que será avaliado
                save_path = os.path.join(args.output_dir, '{}shot_seed{}_repeat{}/inference'.format(shot, seed, repeat_id))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                for factor in args.factors:
                    with open(os.path.join(save_path, 'lvis_instances_results_{}.json'.format(factor)), "w") as f:
                        json.dump(post_preds[str(factor)], f)


def eval():
    args = parse_args()

    cat_ids = [cat['id'] for cat in PACO_NOVEL_CATEGORIES]
    cat_names = [cat['name'] for cat in PACO_NOVEL_CATEGORIES]

    paco_gt = PACO(args.test_dataset_dir)

    for repeat_id in [0]: ## for repeat_id in range(10):
        for shot in [1,2,3,5,10,30]:
            for seed in [0]:
                for factor in args.factors:
                    print("EVAL REPEAT {} SHOT {} SEED {} - FACTOR {}".format(repeat_id, shot, seed, factor))

                    dets_path = os.path.join(args.output_dir, '{}shot_seed{}_repeat{}/inference/lvis_instances_results_{}.json'.format(shot, seed, repeat_id, factor))
                    dets = json.load(open(dets_path))

                    res = PACOResults(paco_gt, dets, max_dets=300)
                    paco_eval = PACOEval(paco_gt, res, iou_type="bbox", attr_ap_type="usual", cat_ids=cat_ids)

                    paco_eval.run()

                    results = paco_eval.get_results()
                    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]

                    # report AP
                    obj_results = dict()
                    for metric in metrics:
                        obj_results[metric] = float(results[metric] * 100)
                    print(obj_results)

                    # report AP for attrs
                    attr_results = heirachrichal_APs(paco_eval)
                    print("AP attr: ", attr_results['obj-attr-AP'])
                    print("AP attr per type: ", attr_results['obj-attrs'], '\n')

def main():
    postprocess()
    eval()

main()