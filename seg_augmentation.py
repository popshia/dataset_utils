import pprint
import yaml
from alive_progress import alive_bar


def load_hyp(hyp):
    with open(hyp) as input:
        hyps = yaml.load(input, Loader=yaml.SafeLoader)

    pprint.pprint(", ".join(f"{key}={value}" for key, value in hyps.items()))
    print("-" * os.get_terminal_size().columns)
    return hyps


def set_up_augseq(hyp):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    return iaa.Sequential(
        [
            # execute 0 to 5 of the following (less important) augmenters per image
            # iaa.SomeOf((0, 5), []),
            # flip
            iaa.Fliplr(hyp["fliplr"]),
            iaa.Flipud(hyp["flipud"]),
            # crop images by -5% to 10% of their height/width
            # iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255)),
            iaa.Affine(
                scale=(1 - hyp["scale"], 1 + hyp["scale"]),
                translate_percent={
                    "x": (-hyp["translate"], hyp["translate"]),
                    "y": (-hyp["translate"], hyp["translate"]),
                },
                rotate=(-hyp["degrees"], hyp["degrees"]),
                shear=(-hyp["shear"], hyp["shear"]),
            ),
            # hsv
            sometimes(
                [
                    iaa.MultiplyBrightness((1 - hyp["hsv_v"], 1 + hyp["hsv_v"])),
                    iaa.MultiplyHue((1 - hyp["hsv_h"], 1 + hyp["hsv_h"])),
                    iaa.MultiplySaturation((1 - hyp["hsv_s"], 1 + hyp["hsv_s"])),
                ]
            ),
            # contrast
            # iaa.GammaContrast((0.5, 2.0)),
            # dropout
            # iaa.Dropout(
            #     (0.01, 0.1), per_channel=0.5
            # ),  # randomly remove up to 10% of the pixels
            # invert
            # iaa.Invert(0.25, per_channel=True),  # invert color channels
        ],
        random_order=True,
    )


import os
import json
import cv2
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.kps import  Keypoint, KeypointsOnImage


def load_yolo_annotations(annotation_file):
    """讀取 YOLOv8 物件分割標註資料"""
    annotations = []
    with open(annotation_file, 'r') as f:
        for line in f.readlines():
            values = list(map(float, line.strip().split()))
            class_id = int(values[0])
            points = np.array(values[1:]).reshape(-1, 2)  # 每兩個值為一組 (x, y)
            annotations.append((class_id, points))
    return annotations

def load_coco_annotations(annotation_file):
    """讀取 COCO 標註資料"""
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    return data

def apply_augmentation(image, polygons, augmenters):
    """對影像和標註的分割點同步進行增強處理"""
    h, w = image.shape[:2]

    # 儲存每個 polygon 的長度
    point_counts = [len(poly) for poly in polygons]
    keypoints = [Keypoint(x * w, y * h) for poly in polygons for x, y in poly]
    kps_on_image = KeypointsOnImage(keypoints, shape=image.shape)
    image_aug, keypoints_aug = augmenters(image=image, keypoints=kps_on_image)
    image_after = keypoints_aug.draw_on_image(image_aug, size=7)
    image_after_rz = cv2.resize(image_after, None, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # 將 keypoints 還原成多個 polygon 轉換回標註格式 (正規化)
    polygons_aug = []
    idx = 0
    for count in point_counts:
        poly = [(keypoints_aug.keypoints[i].x / w, keypoints_aug.keypoints[i].y / h) for i in range(idx, idx + count)]
        polygons_aug.append(poly)
        idx += count

    return image_aug, polygons_aug


def clip_polygon_to_image_border(polygon, img_w, img_h):
    """
    將多邊形限制在影像邊界內，超出範圍的點會被移除或調整到邊界
    以0-indexing處理
    """

    max_x = (img_w - 1) / img_w
    max_y = (img_h - 1) / img_h
    clipped_polygon = []
    for x, y in polygon:
        x = min(max(x, 0), max_x)
        y = min(max(y, 0), max_y)
        clipped_polygon.append((x, y))
    return clipped_polygon


def save_augmented_data(image, annotations, output_img_path, output_ann_path, CLASS_LIST):
    """儲存擴增後的影像與標註"""
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)

    cv2.imencode('.'+output_img_path.split('.')[-1], image)[1].tofile(output_img_path)

    with open(output_ann_path, 'w') as f:
        for class_id, poly in annotations:
            clipped_poly = clip_polygon_to_image_border(poly, image.shape[1], image.shape[0])
            if class_id == CLASS_LIST.index('land'):  # land只要輪廓有超過3個點就保留
                if len(clipped_poly) >= 3:
                    line = f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in clipped_poly]) + "\n"
                    f.write(line)
            else:
                if all(0 <= x <= 1 and 0 <= y <= 1 for x, y in poly):  # 只在所有點都在影像內時保留
                    line = f"{class_id} " + " ".join([f"{x:.6f} {y:.6f}" for x, y in poly]) + "\n"
                    f.write(line)

    print(f"Saved: {output_img_path}, {output_ann_path}")


def main(image_dir, annotation_dir, output_dir, hyp_dir, NEW_IMAGE_TO_CREATE, CLASS_LIST):
    os.makedirs(output_dir, exist_ok=True)

    hyps = load_hyp(hyp_dir)
    augmenters = set_up_augseq(hyps)

    with alive_bar(len(os.listdir(image_dir))) as bar:
        for img_file in os.listdir(image_dir):

            bar.text(os.path.basename(img_file))

            if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(image_dir, img_file)
            ann_path = os.path.join(annotation_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt'))

            if not os.path.exists(ann_path):
                print(f"Warning: No annotation found for {img_file}")
                continue

            # image = cv2.imread(img_path)
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue

            annotations = load_yolo_annotations(ann_path)
            class_ids, polygons = zip(*annotations) if annotations else ([], [])

            for ver in range(1, NEW_IMAGE_TO_CREATE+1):
                image_aug, polygons_aug = apply_augmentation(image, polygons, augmenters)

                augmented_annotations = list(zip(class_ids, polygons_aug))

                img_name, img_ext = os.path.splitext(os.path.basename(img_file))
                output_img_path = os.path.join(output_dir, img_name + f"_aug_{ver}" + img_ext)
                ann_file_name, ann_file_ext = os.path.splitext(os.path.basename(ann_path))
                output_ann_path = os.path.join(output_dir, ann_file_name + f'_aug_{ver}' + ann_file_ext)

                save_augmented_data(image_aug, augmented_annotations, output_img_path, output_ann_path, CLASS_LIST)

            bar()


# if __name__ == "__main__":
#     IMAGE_DIR = r"D:\c-link\個人區\Tai\專案區\114-0004-萬海智慧監控二期\Tai\程式撰寫\訓練資料\標註檔\20250507_6cls_modify_aug_land_label\YOLODataset\train\images"
#     ANNOTATION_DIR = r"D:\c-link\個人區\Tai\專案區\114-0004-萬海智慧監控二期\Tai\程式撰寫\訓練資料\標註檔\20250507_6cls_modify_aug_land_label\YOLODataset\train\labels"
#     OUTPUT_DIR = r"D:\c-link\個人區\Tai\專案區\114-0004-萬海智慧監控二期\Tai\程式撰寫\訓練資料\標註檔\20250507_6cls_modify_aug_land_label\YOLODataset\train\images\aug"
#     HYP_DIR = r"D:\c-link\個人區\Tai\專案區\114-0004-萬海智慧監控二期\Tai\程式撰寫\訓練資料\標註檔\20250507_6cls_modify_aug_land_label\YOLODataset\aug_hyp.yaml"
#     NEW_IMAGE_TO_CREATE = 5
#     CLASS_LIST = ['main_vessel', 'land', 'big_vessel', 'waypoint', 'last_waypoint', 'small_vessel']
#     main(IMAGE_DIR, ANNOTATION_DIR, OUTPUT_DIR, HYP_DIR, NEW_IMAGE_TO_CREATE, CLASS_LIST)

if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("image-dir")
    parser.add_argument("annotation-dir")
    parser.add_argument("output-dir")
    parser.add_argument("hyp")
    parser.add_argument("classes")
    parser.add_argument("--new-image", type=int, default=5)
    args = parser.parse_args()
    main(args.image_dir, args.annotation_dir, args.output_dir, args.hyp, args.new_image, args.classes)
    print(f"done in {time.time() - start:.2f} seconds.")
