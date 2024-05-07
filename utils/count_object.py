import argparse
from pathlib import Path


def count_dataset_objects(args):
    count = 0

    for txt in Path(args.dataset_dir).glob("**/*.txt"):
        with open(txt, "r") as label:
            while label.readline():
                count = count + 1

    print("object count:", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    args = parser.parse_args()
    count_dataset_objects(args)
