import argparse
import os
import pprint
import time
from pathlib import Path

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import yaml
from alive_progress import alive_bar
from imgaug.augmentables.batches import Batch
from imgaug.augmentables.bbs import BoundingBoxesOnImage

from utils.data_augmentation_utils import xywh2xyxy
from utils.general import xyxy2xywh


def load_label(path):
    """
    從 path 讀取標籤檔，若遇到行數不符、內容無法轉 float 等情況，
    會印出警告並跳過該行。
    """
    label_data = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    
    for line_num, line in enumerate(lines):
        # 如果行是空的，直接跳過
        if not line.strip():
            print(f"[Warning] {path} 第 {line_num+1} 行是空白，已跳過。")
            continue
        
        parts = line.split()
        # 這裡檢查欄位數量是否正確 (YOLO 預設5個: class, x, y, w, h)
        if len(parts) != 5:
            print(f"[Warning] {path} 第 {line_num+1} 行欄位數 != 5, 內容: {line}")
            # 根據需求：要直接跳過或 raise Error?
            # 這裡示範跳過
            continue
        
        # 嘗試把每個欄位轉成 float
        try:
            parts = [float(x) for x in parts]
        except ValueError:
            print(f"[Warning] {path} 第 {line_num+1} 行無法轉成 float, 內容: {line}")
            continue
        
        label_data.append(parts)
    
    # 將整理好的 list 轉成 NumPy array
    labels = np.array(label_data, dtype=np.float32)
    return labels



def load_hyp(hyp):
    with open(hyp) as input:
        hyps = yaml.load(input, Loader=yaml.SafeLoader)  # load hyps

    pprint.pprint(", ".join(f"{k}={v}" for k, v in hyps.items()))
    print("-" * os.get_terminal_size().columns)

    return hyps


def label_to_ia_bbx(labels, shape):
    bbxs = []
    for label in labels:
        bbxs.append(
            ia.BoundingBox(
                x1=label[1] * shape[1],
                y1=label[2] * shape[0],
                x2=label[3] * shape[1],
                y2=label[4] * shape[0],
                label=int(label[0]),
            )
        )
    return BoundingBoxesOnImage(bbxs, shape=shape)


def setup_augseq(hyp):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    return iaa.Sequential(
        [
            iaa.Fliplr(hyp["fliplr"]),
            iaa.Flipud(hyp["flipud"]),
            iaa.Affine(
                scale={"x": (1 - hyp["scale"], 1 + hyp["scale"]),
                       "y": (1 - hyp["scale"], 1 + hyp["scale"])},
                translate_percent={"x": (-hyp["translate"], hyp["translate"]),
                                   "y": (-hyp["translate"], hyp["translate"])},
                rotate=(-hyp["degrees"], hyp["degrees"]),
                shear=(-hyp["shear"], hyp["shear"]),
            ),
            # HSV（保留你的設定）
            sometimes([
                iaa.MultiplyBrightness((1 - hyp["hsv_v"], 1 + hyp["hsv_v"])),
                iaa.MultiplyHue((1 - hyp["hsv_h"], 1 + hyp["hsv_h"])),
                iaa.MultiplySaturation((1 - hyp["hsv_s"], 1 + hyp["hsv_s"])),
            ]),

            # === 強對比 / 高光陰影：OneOf 隨機擇一 ===
            iaa.Sometimes(
                hyp.get("contrast_p", 0.5),
                iaa.OneOf([
                    # 對比（全域）
                    iaa.GammaContrast((1 - hyp["contrast"], 1 + hyp["contrast"])),
                    iaa.LinearContrast((1 - hyp["contrast"], 1 + hyp["contrast"])),

                    # 更強勢的明暗拉伸（中灰 S 型）
                    iaa.SigmoidContrast(
                        gain=(5, 12),          # gain 越大對比越強，可依需求再放大
                        cutoff=(0.4, 0.6)      # 中點，控制高光/陰影哪邊更吃重
                    ),

                    # 局部對比（強調高光/陰影細節）
                    iaa.CLAHE(
                        clip_limit=(1.0, hyp.get("clahe_clip", 2.0)),
                        tile_grid_size_px=(8, 8)  # 常見預設，可改成 (4,16) 範圍
                    ),

                    # 亮度乘+加，與對比合用能做頂亮壓暗效果
                    iaa.MultiplyAndAddToBrightness(
                        mul=(1 - hyp["contrast"], 1 + hyp["contrast"]),
                        add=(
                            -int(30 * hyp["contrast"]),
                             int(30 * hyp["contrast"])
                        )
                    ),
                ])
            ),
        ],
        random_order=True,
    )


def read_images(images):
    org_images = []
    for image in images:
        org_images.append(cv2.cvtColor(cv2.imread(image.as_posix()), cv2.COLOR_BGR2RGB))
    return org_images


def read_labels(txts, images):  # images: List[np.ndarray]
    org_bbs = []
    for i, txt in enumerate(txts):
        labels = load_label(txt)
        if labels.shape[0] != 0:
            labels[:, 1:] = xywh2xyxy(labels[:, 1:])

        image_shape = images[i].shape  # (H, W, C)
        org_bbs.append(label_to_ia_bbx(labels, image_shape))
    return org_bbs


def split_batches(list, batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i : i + batch_size]


def create_generator(list):
    for list_entry in list:
        yield list_entry


def save_aug_img_and_label(aug_img, aug_labels, path, batch, ver):
    # create dir
    if not Path("runs/augmentation").is_dir():
        Path("runs/augmentation").mkdir(parents=True, exist_ok=True)

    # create img and txt path
    output_img_name = "runs/augmentation/" + path.stem + f"_{batch}_{ver}" + path.suffix
    output_txt_name = "runs/augmentation/" + path.stem + f"_{batch}_{ver}.txt"
    w, h = aug_img.shape[1], aug_img.shape[0]

    # show augmented image
    # ia.imshow(aug_labels.draw_on_image(aug_img, size=1))

    # write aug_label to txt
    with open(output_txt_name, "w") as txt:
        for box in aug_labels:
            center_x, center_y = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
            if 0 < center_x < w and 0 < center_y < h:
                box.x1, box.x2 = box.x1.clip(0, w), box.x2.clip(0, w)
                box.y1, box.y2 = box.y1.clip(0, h), box.y2.clip(0, h)
                xywh = xyxy2xywh((box.x1, box.y1, box.x2, box.y2))
                xywh /= (w, h, w, h)
                line = f"{int(box.label)} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n"
                txt.writelines(line)

    # save aug_image
    cv2.imwrite(output_img_name, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))


def aug_img(dataset, seq, new_image_count, no_background):
    # 設定 batch 大小
    number_of_batch = 2

    # 取得影像檔案列表
    files = set()
    files.update(Path(dataset).glob("**/*.[jJ][pP][gG]"))
    files.update(Path(dataset).glob("**/*.[jJ][pP][eE][gG]"))
    files.update(Path(dataset).glob("**/*.[bB][mM][pP]"))
    files.update(Path(dataset).glob("**/*.[pP][nN][gG]"))
    image_list = sorted(list(files))
    
    # 取得標籤 txt 檔案列表
    label_list = sorted(list(Path(dataset).glob("**/*.txt")))
    
    # 檢查每個影像檔與標籤檔是否成對出現（以檔名 stem 比對）
    image_dict = {p.stem: p for p in image_list}
    label_dict = {p.stem: p for p in label_list}

    missing_txt = []
    missing_image = []
    for stem, img_path in image_dict.items():
        if stem not in label_dict:
            missing_txt.append(img_path)
    for stem, label_path in label_dict.items():
        if stem not in image_dict:
            missing_image.append(label_path)

    if missing_txt or missing_image:
        print("發現資料不齊全：")
        if missing_txt:
            print("缺少標籤的影像（僅有影像檔，找不到對應的 txt）：")
            for p in missing_txt:
                print(f"  {p}")
        if missing_image:
            print("缺少影像的標籤（僅有 txt 檔，找不到對應的影像）：")
            for p in missing_image:
                print(f"  {p}")
        # 根據需求，可選擇結束程式或僅處理有對應資料的部分
        # exit(0)

    # 僅保留同時擁有影像與標籤的檔案（根據檔名比對）
    matched_stems = set(image_dict.keys()) & set(label_dict.keys())
    image_list = sorted([image_dict[stem] for stem in matched_stems])
    label_list = sorted([label_dict[stem] for stem in matched_stems])

    # 以下原有程式碼：分批處理資料等
    # 分批處理
    image_list_batches = list(split_batches(image_list, number_of_batch))
    label_list_batches = list(split_batches(label_list, number_of_batch))
    
    # 建立增強批次
    aug_batch = []
    print("reading images and labels to memory and split batches...")
    for i in range(len(image_list_batches)):
        images = read_images(image_list_batches[i])
        labels = read_labels(label_list_batches[i], images)
        aug_batch.append(
            [
                Batch(images=images, bounding_boxes=labels, data=image_list_batches[i])
                for _ in range(new_image_count)
            ]
        )
        
    # 開始進行增強並儲存結果
    with alive_bar(len(aug_batch)) as bar:
        print("augmenting batch by batch...")
        for batch_num, batch in enumerate(aug_batch):
            auged_batch = seq.augment_batches(batch, background=no_background)
            for ver_num, aug in enumerate(auged_batch):
                for i, image in enumerate(aug.images_aug):
                    bbs = aug.bounding_boxes_aug[i]
                    path = aug.data[i]
                    bar.text(path.stem)
                    save_aug_img_and_label(image, bbs, path, batch_num + 1, ver_num + 1)
            bar()


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp")
    parser.add_argument("dataset")
    parser.add_argument("--new-image", type=int, default=5)
    parser.add_argument("--no-background", action="store_false")
    args = parser.parse_args()

    dataset = args.dataset
    new_image_count = args.new_image
    hyps = load_hyp(args.hyp)
    seq = setup_augseq(hyps)

    aug_img(dataset, seq, new_image_count, args.no_background)
    print(f"done in {time.time() - start:.2f} seconds.")
