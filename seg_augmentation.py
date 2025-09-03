import argparse
import json
import os
import pprint
import time

import cv2
import imgaug.augmenters as iaa
import numpy as np
import yaml
from alive_progress import alive_bar
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage


def load_hyp(hyp_path):
    with open(hyp_path) as file:
        hyp = yaml.load(file, Loader=yaml.SafeLoader)
    pprint.pprint(", ".join(f"{k}={v}" for k, v in hyp.items()))
    print("-" * os.get_terminal_size().columns)
    return hyp


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


def load_yolo_annotations(file_path):
    annotations = []
    with open(file_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls_id = int(parts[0])
            points = np.array(parts[1:]).reshape(-1, 2)
            annotations.append((cls_id, points))
    return annotations


def load_coco_annotations(annotation_file):
    """讀取 COCO 標註資料"""
    with open(annotation_file, "r") as f:
        data = json.load(f)
    return data


def apply_augmentation(image, polygons, augmenter):
    h, w = image.shape[:2]
    point_counts = [len(poly) for poly in polygons]
    keypoints = [Keypoint(x * w, y * h) for poly in polygons for x, y in poly]
    kps = KeypointsOnImage(keypoints, shape=image.shape)
    image_aug, kps_aug = augmenter(image=image, keypoints=kps)

    idx = 0
    polygons_aug = []
    for count in point_counts:
        poly = [
            (kps_aug.keypoints[i].x / w, kps_aug.keypoints[i].y / h)
            for i in range(idx, idx + count)
        ]
        polygons_aug.append(poly)
        idx += count

    return image_aug, polygons_aug


def correct_point_to_boundary(cx, cy, x, y, img_w, img_h):
    dx = x - cx
    dy = y - cy

    if dx == 0 and dy == 0:
        return cx, cy

    intersections = []

    # 與左邊界 (x = 0)
    if dx != 0:
        t = -cx / dx
        py = cy + t * dy
        if t >= 0 and 0 <= py <= img_h - 1:
            intersections.append((0, py))

    # 與右邊界 (x = img_w - 1)
    if dx != 0:
        t = (img_w - 1 - cx) / dx
        py = cy + t * dy
        if t >= 0 and 0 <= py <= img_h - 1:
            intersections.append((img_w - 1, py))

    # 與上邊界 (y = 0)
    if dy != 0:
        t = -cy / dy
        px = cx + t * dx
        if t >= 0 and 0 <= px <= img_w - 1:
            intersections.append((px, 0))

    # 與下邊界 (y = img_h - 1)
    if dy != 0:
        t = (img_h - 1 - cy) / dy
        px = cx + t * dx
        if t >= 0 and 0 <= px <= img_w - 1:
            intersections.append((px, img_h - 1))

    if intersections:
        # 回傳距離起點最近的交點
        intersections.sort(key=lambda pt: (pt[0] - cx) ** 2 + (pt[1] - cy) ** 2)
        return intersections[0]

    return cx, cy


def save_augmented(image, annotations, img_out_path, ann_out_path):
    os.makedirs(os.path.dirname(img_out_path), exist_ok=True)
    cv2.imencode("." + img_out_path.split(".")[-1], image)[1].tofile(img_out_path)

    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    with open(ann_out_path, "w") as f:
        for cls_id, poly in annotations:
            corrected_norm = []
            for x_norm, y_norm in poly:
                # 轉為像素座標
                x_pix = x_norm * w
                y_pix = y_norm * h
                # 如果超出邊界，修正到邊界
                if not (0 <= x_pix <= w - 1 and 0 <= y_pix <= h - 1):
                    x_corr, y_corr = correct_point_to_boundary(
                        cx, cy, x_pix, y_pix, w, h
                    )
                    print(f"修正點 ({x_pix}, {y_pix}) 到邊界 ({x_corr}, {y_corr})")
                else:
                    x_corr, y_corr = x_pix, y_pix
                # 轉回歸一化座標
                corrected_norm.append((x_corr / w, y_corr / h))

            line = (
                f"{cls_id} "
                + " ".join(f"{x:.6f} {y:.6f}" for x, y in corrected_norm)
                + "\n"
            )
            f.write(line)
    print(f"Saved: {img_out_path}, {ann_out_path}")


def process_dataset(
    image_dir, annotation_dir, output_dir, hyp_file, num_augs, class_list
):
    os.makedirs(output_dir, exist_ok=True)
    hyp = load_hyp(hyp_file)
    augmenter = set_up_augseq(hyp)
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    with alive_bar(len(image_files)) as bar:
        for img_file in image_files:
            try:
                bar.text(img_file)
                img_path = os.path.join(image_dir, img_file)
                ann_path = os.path.join(
                    annotation_dir, os.path.splitext(img_file)[0] + ".txt"
                )

                if not os.path.exists(ann_path):
                    print(f"Warning: No annotation found for {img_file}")
                    bar()
                    continue

                image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)

                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # remove alpha

                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    bar()
                    continue

                annotations = load_yolo_annotations(ann_path)
                class_ids, polygons = zip(*annotations) if annotations else ([], [])

                for i in range(1, num_augs + 1):
                    img_aug, poly_aug = apply_augmentation(image, polygons, augmenter)
                    aug_ann = list(zip(class_ids, poly_aug))

                    base_name, ext = os.path.splitext(img_file)
                    img_out = os.path.join(output_dir, f"{base_name}_aug_{i}{ext}")
                    ann_out = os.path.join(output_dir, f"{base_name}_aug_{i}.txt")

                    save_augmented(img_aug, aug_ann, img_out, ann_out)

            except Exception as e:
                print(f"Error: {e}, {img_path}, {ann_path}")
                return
            bar()


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir")
    parser.add_argument("annotation_dir")
    parser.add_argument("output_dir")
    parser.add_argument("hyp")
    parser.add_argument("--new_image", type=int, default=5)
    args = parser.parse_args()
    classes = [""]
    try:
        process_dataset(
            args.image_dir,
            args.annotation_dir,
            args.output_dir,
            args.hyp,
            args.new_image,
            classes,
        )
        print(f"done in {time.time() - start:.2f} seconds.")
    except Exception as e:
        print(f"Error: {e}")
