import argparse
from pathlib import Path

IMG_TYPE_PATTERN = [
    "**/*.jpeg",
    "**/*.jpg",
    "**/*.JPG",
    "**/*.png",
    "**/*.PNG",
    "**/*.bmp",
    "**/*.BMP",
]


def replenish_empty_txt(dataset):
    img_list = []
    for pattern in IMG_TYPE_PATTERN:
        img_list.extend(Path(dataset).glob(pattern))

    img_list = sorted(img_list)
    txt_list = sorted(Path(dataset).glob("**/*.txt"))
    for img in img_list:
        if img.with_suffix(".txt") not in txt_list:
            print(img.with_suffix(".txt"))
            open(img.with_suffix(".txt"), "w")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset directory")
    args = parser.parse_args()
    replenish_empty_txt(args.dataset)
