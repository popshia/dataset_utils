import argparse
from pathlib import Path


def generate_txt(args):
    train_dir = Path(args.train_dir).resolve()
    val_dir = Path(args.val_dir).resolve()
    train_txt = Path(train_dir).parents[0] / "train.txt"
    val_txt = Path(val_dir).parents[0] / "val.txt"

    # train dir
    with open(train_txt, "w") as train_txt:
        for img in sorted(Path(train_dir).glob("**/*.[jJpP][pPnN][gG]")):
            img_path = Path(img).resolve()
            train_txt.write(str(img_path) + "\n")

    # val dir
    with open(val_txt, "w") as val_txt:
        for img in sorted(Path(val_dir).glob("**/*.[jJpP][pPnN][gG]")):
            img_path = Path(img).resolve()
            val_txt.write(str(img_path) + "\n")

    # test dir
    if args.test_dir:
        test_dir = Path(args.test_dir).resolve()
        test_txt = Path(test_dir).parents[0] / "test.txt"

        with open(test_txt, "w") as test_txt:
            for img in sorted(Path(test_dir).glob("**/*.[jJpP][pPnN][gG]")):
                img_path = Path(img).resolve()
                test_txt.write(str(img_path) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_dir", help="training dataset directory")
    parser.add_argument("val_dir", help="validation dataset directory")
    parser.add_argument("--test-dir", help="test dataset directory")
    args = parser.parse_args()
    generate_txt(args)
