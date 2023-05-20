import argparse
import json
import os
import random


def generate_seeds():
    data_path = "/data/datasets/paco/annotations/paco_lvis_v1_train.json"
    data = json.load(open(data_path))

    id2img = {} # diccionario de diccionarios de imaxes indexado polo id da imaxe
    for i in data["images"]:
        id2img[i["id"]] = i

    anno = {i: [] for i in ID2CLASS.keys()} # diccionario de listas de anotacións indexado por id de categoría

    # Incluir as anotacións na lista de anotacións da categoría correspondente
    for a in data["annotations"]:
        # if a["iscrowd"] == 1: # ignorar anotacións crowd de COCO
        #     continue
        if a["category_id"] in ID2CLASS.keys():
            anno[a["category_id"]].append(a)

    # Iterar sobre as 10 seeds
    for i in range(10):
        random.seed(i)

        # Iterar sobre as categorías
        for c in ID2CLASS.keys():
            img_ids = {} # diccionario de listas de anotacións indexado por id da imaxe

            # Insertar as anotacións da categoría en img_ids
            for a in anno[c]:
                if a["image_id"] in img_ids:
                    img_ids[a["image_id"]].append(a) 
                else:
                    img_ids[a["image_id"]] = [a]

            sample_shots = [] # lista de instancias do shot
            sample_imgs = []  # lista de imaxes do shot
            for shots in [1, 2, 3, 5, 10, 30]:
                while True:
                    imgs = random.sample(list(img_ids.keys()), shots) # lista de <shots> ids de imaxe aleatorios (con anotacións da categoría)
                    for img in imgs:
                        skip = False
                        # Se xa incluimos as anotacións da imaxe na lista de instancias (sample_shots), pasamos á seguinte imaxe
                        for s in sample_shots:
                            if img == s["image_id"]:
                                skip = True
                                break
                        if skip:
                            continue

                        # Se o num de anotacións na imaxe + o num de instancias xa seleccionadas é maior que <shots>, pasamos á seguinte imaxe
                        if len(img_ids[img]) + len(sample_shots) > shots:
                            continue

                        # Engadimos á lista de instancias (sample_shots) todas as anotacións da imaxe
                        sample_shots.extend(img_ids[img])
                        # Incluimos o diccionario da imaxe a sample_imgs
                        sample_imgs.append(id2img[img])
                        if len(sample_shots) == shots:
                            break
                    if len(sample_shots) == shots:
                        break
                
                new_data = {
                    "images": sample_imgs,
                    "annotations": sample_shots,
                    "categories": data["categories"], # todas as categorías (base e novel)
                    "attributes": data["attributes"], # todos os atributos
                    "attr_type_to_attr_idxs": data["attr_type_to_attr_idxs"],
                }
                save_path = get_save_path_seeds(
                    data_path, ID2CLASS[c], shots, i
                )
                with open(save_path, "w") as f:
                    json.dump(new_data, f)


def get_save_path_seeds(path, cls, shots, seed):
    # prefix = "full_box_{}shot_{}_trainval".format(shots, cls)
    prefix = "full_box_{}shot_{}_train".format(shots, cls)
    save_dir = os.path.join("/data", "datasets", "pacosplit", "seed" + str(seed))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, prefix + ".json")
    return save_path


if __name__ == "__main__":
    # Diccionario coas clases novel
    ID2CLASS = {
        # 23: "trash_can",
        # 35: "handbag",
        41: "ball",
        # 61: "basket",
        # 88: "belt",
        # 90: "bench",
        94: "bicycle",
        # 112: "blender",
        # 127: "book",
        # 133: "bottle",
        # 139: "bowl",
        # 143: "box",
        # 156: "broom",
        # 160: "bucket",
        # 184: "calculator",
        # 192: "can",
        207: "car_(automobile)",
        # 220: "carton",
        # 230: "cellular_telephone",
        # 232: "chair",
        271: "clock",
        # 324: "crate",
        # 344: "cup",
        378: "dog",
        # 396: "drill",
        # 399: "drum_(musical_instrument)",
        409: "earphone",
        429: "fan",
        # 498: "glass_(drink_container)",
        # 521: "guitar",
        # 530: "hammer",
        # 544: "hat",
        # 556: "helmet",
        # 591: "jar",
        # 604: "kettle",
        615: "knife",
        621: "ladder",
        # 626: "lamp",
        # 631: "laptop_computer",
        687: "microwave_oven",
        694: "mirror",
        705: "mouse_(computer_equipment)",
        # 708: "mug",
        # 713: "napkin",
        719: "newspaper",
        # 751: "pan_(for_cooking)",
        # 781: "pen",
        # 782: "pencil",
        # 804: "pillow",
        # 811: "pipe",
        # 818: "plate",
        # 821: "pliers",
        # 881: "remote_control",
        # 898: "plastic_bag",
        921: "scarf",
        # 923: "scissors",
        # 926: "screwdriver",
        # 948: "shoe",
        # 973: "slipper_(footwear)",
        979: "soap",
        # 999: "sponge",
        1000: "spoon",
        # 1018: "stool",
        1042: "sweater",
        1050: "table",
        # 1061: "tape_(sticky_cloth_or_paper)",
        # 1072: "telephone",
        # 1077: "television_set",
        # 1093: "tissue_paper",
        1108: "towel",
        # 1117: "tray",
        # 1139: "vase",
        # 1156: "wallet",
        1161: "watch",
        # 1196: "wrench",
    }
    CLASS2ID = {v: k for k, v in ID2CLASS.items()}


    generate_seeds()