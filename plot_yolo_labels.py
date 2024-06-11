import argparse
import random
import threading
from pathlib import Path

import cv2


def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    tl = (
        line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def draw_box_on_image(img_path, label_path, classes, colors, img_save_dir):
    save_img_path = Path(img_save_dir) / Path(img_path).parts[-1]

    label = open(label_path) if Path(label_path).exists() else []
    image = cv2.imread(img_path)
    try:
        height, width, channels = image.shape
    except:
        print("no shape info.")
        return 0

    cls_count = {}
    for c in classes:
        cls_count[c] = 0

    box_number = 0
    for line in label:  # iterate txt lines
        staff = line.split()  # split attributes
        class_idx = int(staff[0])
        cls_count[classes[class_idx]] += 1

        x_center, y_center, w, h = (
            float(staff[1]) * width,
            float(staff[2]) * height,
            float(staff[3]) * width,
            float(staff[4]) * height,
        )
        x1 = round(x_center - w / 2)
        y1 = round(y_center - h / 2)
        x2 = round(x_center + w / 2)
        y2 = round(y_center + h / 2)

        plot_one_box(
            [x1, y1, x2, y2],
            image,
            color=colors[class_idx],
            label=classes[class_idx],
            line_thickness=None,
        )

        if not Path(img_save_dir).exists():
            Path(img_save_dir).mkdir()

        cv2.imwrite(save_img_path.as_posix(), image)

        box_number += 1

    print("\n{}: {} boxes.".format(Path(img_path).parts[-1], box_number))
    for k, v in cls_count.items():
        print("{} {}".format(v, k))
    print("\n", end="")


def plot_yolo_labels(args):
    classes = open(args.classes_txt).read().strip().split("\n")
    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
    thread_list = []

    for i, img in enumerate(
        sorted(Path(args.dataset_dir).glob("*.[jJpP][pPnN][gG]"))
    ):  # iterate img names
        img_path = Path(img).resolve().as_posix()
        label_path = Path(img).with_suffix(".txt").as_posix()
        thread_list.append(
            threading.Thread(
                target=draw_box_on_image,
                args=(img_path, label_path, classes, colors, "./runs/plot"),
            )
        )
        if i == args.output_count - 1:
            break

    for thread in thread_list:
        thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument("classes_txt")
    parser.add_argument("output_count", type=int)
    args = parser.parse_args()
    plot_yolo_labels(args)
