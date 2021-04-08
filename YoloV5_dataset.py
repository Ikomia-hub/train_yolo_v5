import os
import random
import shutil
import yaml


def prepare(ikdataset, dataset_folder, split_ratio):
    # TODO: if source format is already YoloV5 we just have to get folder
    train_img_folder = dataset_folder + os.sep + "images" + os.sep + "train"
    val_img_folder = dataset_folder + os.sep + "images" + os.sep + "val"
    train_label_folder = dataset_folder + os.sep + "labels" + os.sep + "train"
    val_label_folder = dataset_folder + os.sep + "labels" + os.sep + "val"
    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(val_img_folder, exist_ok=True)
    os.makedirs(train_label_folder, exist_ok=True)
    os.makedirs(val_label_folder, exist_ok=True)

    images = ikdataset.data["images"]
    val_size = int((1-split_ratio) * len(images))
    val_indices = random.sample(range(len(images)), k=val_size)
    index = 0

    for img in images:
        src_filename = img["filename"]

        if index in val_indices:
            dst_filename = val_img_folder + os.sep + os.path.basename(src_filename)
            shutil.copy(src_filename, dst_filename)
            dst_filename = dst_filename.replace("images" + os.sep + "val", "labels" + os.sep + "val", 1)
            dst_filename = dst_filename.replace('.' + dst_filename.split('.')[-1], '.txt')
            _create_image_labels(dst_filename, img["annotations"], img["width"], img["height"])
        else:
            dst_filename = train_img_folder + os.sep + os.path.basename(src_filename)
            shutil.copy(src_filename, dst_filename)
            dst_filename = dst_filename.replace("images" + os.sep + "train", "labels" + os.sep + "train", 1)
            dst_filename = dst_filename.replace('.' + dst_filename.split('.')[-1], '.txt')
            _create_image_labels(dst_filename, img["annotations"], img["width"], img["height"])

        index += 1

    categories = ikdataset.data["metadata"]["category_names"]
    return _create_dataset_yaml(dataset_folder, train_img_folder, val_img_folder, categories)


def _create_image_labels(filename, annotations, img_w, img_h):
    with open(filename, "w+") as f:
        for ann in annotations:
            box = ann['bbox']
            cx = (box[0] + (box[2] / 2)) / img_w
            cy = (box[1] + (box[3] / 2)) / img_h
            width = box[2] / img_w
            height = box[3] / img_h
            f.write('%d ' % ann["category_id"])
            f.write('%f ' % cx)
            f.write('%f ' % cy)
            f.write('%f ' % width)
            f.write('%f \r\n' % height)


def _create_dataset_yaml(dataset_folder, train_folder, val_folder, categories):
    dataset = {"train": train_folder,
               "val": val_folder,
               "nc": len(categories),
               "names": list(categories.values())}

    dataset_yaml_file = dataset_folder + os.sep + "dataset.yaml"
    with open(dataset_yaml_file, "w") as f:
        yaml.dump(dataset, f, default_flow_style=True, sort_keys=False)

    return dataset_yaml_file

