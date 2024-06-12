import argparse
import math
import os
import pprint
import random
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml

from utils.data_augmentation_utils import (
    box_candidates,
    resample_segments,
    segment2box,
    xywhn2xyxy,
    xyxy2xywh,
)


def letterbox(
    img,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return img, ratio, (dw, dh)


def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


def load_hyp(hyp):
    with open(hyp) as input:
        hyps = yaml.load(input, Loader=yaml.SafeLoader)  # load hyps

    return hyps


def load_image(args, path):
    img = cv2.imread(path)  # BGR
    assert img is not None, "Image Not Found: " + path
    h0, w0 = img.shape[:2]  # orig hw
    r = args.img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        img = cv2.resize(
            img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR
        )
    return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized


def load_label(path):
    with open(path, "r") as input:
        labels = [x.split() for x in input.read().strip().splitlines()]
        labels = np.array(labels, dtype=np.float32)

    return labels


def random_perspective(
    img,
    targets=(),
    segments=(),
    degrees=10,
    translate=0.1,
    scale=0.1,
    shear=10,
    perspective=0.0,
    border=(0, 0),
):
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    # s = random.uniform(1 - scale, 1.1 + scale)
    s = random.uniform(1, 1.1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * width
    )  # x translation (pixels)
    T[1, 2] = (
        random.uniform(0.5 - translate, 0.5 + translate) * height
    )  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

    # org_img = img.copy()

    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(
                img, M, dsize=(width, height), borderValue=(114, 114, 114)
            )
        else:  # affine
            img = cv2.warpAffine(
                img, M[:2], dsize=(width, height), borderValue=(114, 114, 114)
            )

    # # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(org_img[:, :, ::-1])  # base
    # ax[1].imshow(img[:, :, ::-1])  # warped
    # plt.show()

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:  # warp segments
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T  # transform
                xy = (
                    xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
                )  # perspective rescale or affine

                # clip
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(
                n, 8
            )  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            )

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(
            box1=targets[:, 1:5].T * s,
            box2=new.T,
            area_thr=0.01 if use_segments else 0.10,
        )
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return img, targets


def data_augmentation(args, hyp, image_path, i):
    # Load image
    img, (h0, w0), (h, w) = load_image(args, image_path.as_posix())

    # Letterbox
    shape = args.img_size  # final letterboxed shape
    img, ratio, pad = letterbox(img, shape, auto=False, scaleup=True)

    labels = load_label(Path(image_path).with_suffix(".txt"))
    if len(labels) > 0:  # normalized xywh to pixel xyxy format
        labels[:, 1:] = xywhn2xyxy(
            labels[:, 1:],
            int(ratio[0]) * int(w),
            int(ratio[1]) * int(h),
            padw=int(pad[0]),
            padh=int(pad[1]),
        )

    org_img = img.copy()

    img, labels = random_perspective(
        img,
        labels,
        degrees=hyp["degrees"],
        translate=hyp["translate"],
        scale=hyp["scale"],
        shear=hyp["shear"],
        perspective=hyp["perspective"],
    )

    # Augment colorspace
    augment_hsv(img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])

    nL = len(labels)  # number of labels
    if nL:
        labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
        labels[:, [2, 4]] /= img.shape[0]  # normalized height 0-1
        labels[:, [1, 3]] /= img.shape[1]  # normalized width 0-1

    # flip up-down
    if random.random() < hyp["flipud"]:
        img = np.flipud(img)
        if nL:
            labels[:, 2] = 1 - labels[:, 2]

    # flip left-right
    if random.random() < hyp["fliplr"]:
        img = np.fliplr(img)
        if nL:
            labels[:, 1] = 1 - labels[:, 1]

    # Visualize
    # import matplotlib.pyplot as plt
    # from data_augmentation_utils import plot_one_box, xywh2xyxy
    # normalized_label = labels[:, 1:] * (640, 640, 640, 640)
    # normalized_label = xywh2xyxy(normalized_label)
    # print("normalized label\n", normalized_label)
    # img = np.ascontiguousarray(img)
    # for label in normalized_label:
    #     label = np.asarray(label, dtype=int)
    #     plot_one_box(label, img)
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(org_img[:, :, ::-1])  # base
    # ax[1].imshow(img[:, :, ::-1])  # warped
    # plt.show()

    if not Path("runs/augmentation").is_dir():
        Path("runs/augmentation").mkdir(parents=True, exist_ok=True)

    output_img_name = (
        "runs/augmentation" + image_path.stem + "_{}".format(str(i)) + image_path.suffix
    )
    output_txt_name = "runs/augmentation" + image_path.stem + "_{}.txt".format(str(i))
    print(output_img_name)
    # print(output_txt_name)

    with open(output_txt_name, "w") as txt:
        for label in labels:
            label_list = label.tolist()
            line = "{:d} {:6f} {:6f} {:6f} {:6f}\n".format(
                int(label_list[0]),
                label_list[1],
                label_list[2],
                label_list[3],
                label_list[4],
            )
            txt.writelines(line)

    cv2.imwrite(output_img_name, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", help="hyperparameter yaml file.")
    parser.add_argument("dataset", help="dataset directory.")
    parser.add_argument(
        "--new-image",
        type=int,
        default=3,
        help="how many new image to create per image.",
    )
    parser.add_argument("--img-size", type=int, default=640, help="resize image size.")
    args = parser.parse_args()
    hyps = load_hyp(args.hyp)

    pprint.pprint(", ".join(f"{k}={v}" for k, v in hyps.items()))
    print("-" * os.get_terminal_size().columns)
    thread_list = []
    img_list = sorted(Path(args.dataset).glob("**/*.[jJpP][pPnN][gG]"))

    for img in img_list:
        for i in range(args.new_image):
            thread_list.append(
                threading.Thread(
                    target=data_augmentation,
                    args=(args, hyps, img, "aug_ver" + str(i + 1)),
                )
            )

    for thread in thread_list:
        thread.start()
