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
            # sometimes lambda
            # sometimes(),
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


def read_images(images):
    org_images = []
    for image in images:
        org_images.append(cv2.cvtColor(cv2.imread(image.as_posix()), cv2.COLOR_BGR2RGB))
    return org_images


def read_labels(txts, images):
    org_bbs = []
    for i, txt in enumerate(txts):
        labels = load_label(txt)
        if labels.shape[0] != 0:
            labels[:, 1:] = xywh2xyxy(labels[:, 1:])
        org_bbs.append(label_to_ia_bbx(labels, images[i].shape))
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
    print(output_img_name, output_txt_name)
    h, w = aug_img.shape[1], aug_img.shape[0]

    # write aug_label to txt
    with open(output_txt_name, "w") as txt:
        for box in aug_labels:
            xywh = xyxy2xywh((box.x1, box.y1, box.x2, box.y2))
            xywh[0], xywh[2] = xywh[0].clip(0, w), xywh[2].clip(0, w)
            xywh[1], xywh[3] = xywh[1].clip(0, w), xywh[3].clip(0, w)
            xywh /= (w, h, w, h)
            line = f"{int(box.label)} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n"
            txt.writelines(line)

    # save aug_image
    # ia.imshow(labels.draw_on_image(img, size=1))
    cv2.imwrite(output_img_name, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))


def aug_img(dataset, seq, new_image_count, no_background):
    # set batch size
    number_of_batch = 200

    # get file list
    image_list = sorted(Path(dataset).glob("**/*.[jJpP][pPnN][gG]"))
    label_list = sorted(Path(dataset).glob("**/*.txt"))

    # read images and labels
    org_images = read_images(image_list)
    org_bbs = read_labels(label_list, org_images)

    # split images, labels and path to batch
    org_images = list(split_batches(org_images, number_of_batch))
    org_bbs = list(split_batches(org_bbs, number_of_batch))
    org_paths = list(split_batches(image_list, number_of_batch))

    # create aug_batch list
    aug_batch = []

    # create batch for augmentation
    for i, split in enumerate(org_images):
        aug_batch.append(
            [
                Batch(images=split, bounding_boxes=org_bbs[i], data=org_paths[i])
                for _ in range(new_image_count)
            ]
        )

    # start augmentation and save aug_imgs and aug_labels
    with alive_bar(len(aug_batch)) as bar:
        bar.text("augmenting batch by batch...")
        for batch_num, batch in enumerate(aug_batch):
            auged_batch = seq.augment_batches(batch, background=no_background)
            for ver_num, aug in enumerate(auged_batch):
                for i, image in enumerate(aug.images_aug):
                    bbs = aug.bounding_boxes_aug[i]
                    path = aug.data[i]
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
