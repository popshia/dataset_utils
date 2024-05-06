import argparse
from pathlib import Path


def generate_txt(opts):
    train_dir = Path(opts.train_dir).resolve()
    val_dir = Path(opts.val_dir).resolve()
    train_txt = Path(train_dir).parents[0] / "train.txt"
    val_txt = Path(val_dir).parents[0] / "val.txt"

    # train dir
    with open(train_txt, "w") as train_txt:
        for img in sorted(Path(train_dir).glob("**/*.[jp][pn]g")):
            img_path = Path(img).resolve()
            if not opts.aug:
                if not any(
                    substr in str(img) for substr in ["_cot", "_brt", "_ivt", "_sat"]
                ):
                    train_txt.write(str(img_path) + "\n")
            else:
                train_txt.write(str(img_path) + "\n")

    # val dir
    with open(val_txt, "w") as val_txt:
        for img in sorted(Path(val_dir).glob("**/*.[jp][pn]g")):
            img_path = Path(img).resolve()
            if not opts.aug:
                if not any(
                    substr in str(img) for substr in ["_cot", "_brt", "_ivt", "_sat"]
                ):
                    val_txt.write(str(img_path) + "\n")
            else:
                val_txt.write(str(img_path) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help="training dataset directory")
    parser.add_argument("val_dir", help="validation dataset directory")
    parser.add_argument("--aug", action="store_true", help="augmentation flag")
    args = parser.parse_args()
    generate_txt(args)
