import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


class Image:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name.resolve().as_posix())
        self.abs_path = img_name.resolve()
        self.raw_name_with_abs_path = self.abs_path.with_suffix("").as_posix()
        self.label_file = self.abs_path.with_suffix(".txt").as_posix()
        self.ext = img_name.suffix
        self.aug_imgs = []

    def convert_colors(self):
        # brightness
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        brt_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        img_name = self.raw_name_with_abs_path + "_brt" + self.ext
        self.aug_imgs.append([img_name, brt_img])

        # saturation
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        sat_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        img_name = self.raw_name_with_abs_path + "_sat" + self.ext
        self.aug_imgs.append([img_name, sat_img])

        # contrast
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(self.img)
        dummy = dummy * (contrast / 127 + 1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        cot_img = np.uint8(dummy)
        img_name = self.raw_name_with_abs_path + "_cot" + self.ext
        self.aug_imgs.append([img_name, cot_img])

        # invert
        inv_img = cv2.bitwise_not(self.img)
        img_name = self.raw_name_with_abs_path + "_ivt" + self.ext
        self.aug_imgs.append([img_name, inv_img])

    def save_aug_imgs(self):
        for i in range(len(self.aug_imgs)):
            print(
                self.aug_imgs[i][0],
                cv2.imwrite(
                    self.aug_imgs[i][0],
                    self.aug_imgs[i][1],
                ),
            )

    def save_aug_labels(self):
        label_file = self.label_file
        # print(label_file)
        # print(label_file[:-4] + "_brt.txt")
        # print(label_file[:-4] + "_cot.txt")
        # print(label_file[:-4] + "_sat.txt")
        # print(label_file[:-4] + "_ivt.txt")
        print("-" * os.get_terminal_size().columns)
        shutil.copy(
            label_file,
            label_file[:-4] + "_brt.txt",
        )
        shutil.copy(
            label_file,
            label_file[:-4] + "_cot.txt",
        )
        shutil.copy(
            label_file,
            label_file[:-4] + "_sat.txt",
        )
        shutil.copy(
            label_file,
            label_file[:-4] + "_ivt.txt",
        )

    def convert(self):
        self.convert_colors()
        self.save_aug_imgs()
        self.save_aug_labels()


def convert_imgs(opts):
    dataset_dir = opts.dataset_dir

    for img in sorted(Path(dataset_dir).glob("**/*.[jp][pn]g")):
        org_img = Image(Path(img))
        org_img.convert()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", help="dataset directory")
    args = parser.parse_args()
    convert_imgs(args)
