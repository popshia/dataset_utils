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
from imgaug import multicore
from imgaug.augmentables.batches import Batch, UnnormalizedBatch
from imgaug.augmentables.bbs import BoundingBoxesOnImage

from utils.data_augmentation_utils import xywh2xyxy
from utils.general import xyxy2xywh


def load_label(path):
    with open(path, "r") as input:
        labels = [x.split() for x in input.read().strip().splitlines()]
        labels = np.array(labels, dtype=np.float32)

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
                x1=int(label[1] * shape[1]),
                y1=int(label[2] * shape[0]),
                x2=int(label[3] * shape[1]),
                y2=int(label[4] * shape[0]),
                label=int(label[0]),
            )
        )
    return BoundingBoxesOnImage(bbxs, shape=shape)


def setup_augseq(hyp):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    return iaa.Sequential(
        [
            # execute 0 to 5 of the following (less important) augmenters per image
            # iaa.SomeOf((0, 5), []),
            # resize and flip
            iaa.Fliplr(hyp["fliplr"]),  # horizontally flip 50% of all images
            iaa.Flipud(hyp["flipud"]),  # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            # sometimes(
            #     iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=ia.ALL, pad_cval=(0, 255))
            # ),
            iaa.Affine(
                scale={
                    "x": (1 - hyp["scale"], 1 + hyp["scale"]),
                    "y": (1 - hyp["scale"], 1 + hyp["scale"]),
                },
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


def read_image(image):
    return cv2.cvtColor(cv2.imread(image.as_posix()), cv2.COLOR_BGR2RGB)


def read_label(txt, image):
    labels = load_label(txt)
    if labels.shape[0] != 0:
        labels[:, 1:] = xywh2xyxy(labels[:, 1:])
    shape = cv2.imread(image.as_posix()).shape
    return label_to_ia_bbx(labels, shape)


def split_batches(list, batch_size):
    for i in range(0, len(list), batch_size):
        yield list[i : i + batch_size]


def create_generator(list):
    for list_entry in list:
        yield list_entry


def save_aug_img_and_label(aug_img, aug_labels, path, ver):
    # create dir
    if not Path("runs/augmentation").is_dir():
        Path("runs/augmentation").mkdir(parents=True, exist_ok=True)

    # create img and txt path
    output_img_name = "runs/augmentation/" + path.stem + f"_{ver}" + path.suffix
    output_txt_name = "runs/augmentation/" + path.stem + f"_{ver}.txt"
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
    # get file list
    image_list = sorted(Path(dataset).glob("**/*.[jJpP][pPnN][gG]"))
    label_list = sorted(Path(dataset).glob("**/*.txt"))

    with alive_bar(len(image_list)) as bar:
        for i in range(len(image_list)):
            bar.text(image_list[i].stem)
            for ver in range(new_image_count):
                image, label = image_list[i], label_list[i]
                aug_img, aug_label = seq(
                    image=read_image(image), bounding_boxes=read_label(label, image)
                )
                save_aug_img_and_label(aug_img, aug_label, image, ver + 1)
            bar()


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp")
    parser.add_argument("dataset")
    parser.add_argument("--new-image", type=int, default=5)
    args = parser.parse_args()

    dataset = args.dataset
    new_image_count = args.new_image
    hyps = load_hyp(args.hyp)
    seq = setup_augseq(hyps)

    aug_img(dataset, seq, new_image_count, args.no_background)
    print(f"done in {time.time() - start:.2f} seconds.")
