import argparse
import os
import shutil
from pathlib import Path
from random import randrange


def create_dirs_and_move_files(file_list, dest):
    for file in file_list:
        new_path = Path(Path.cwd())
        parents = file.parent.as_posix().split("/")[1:]
        parents.insert(0, dest)
        for parent in parents:
            new_path = new_path.joinpath(parent)

        Path.mkdir(new_path, parents=True, exist_ok=True)
        shutil.copy(file, new_path)

        if Path(file.with_suffix(".txt")).exists():
            shutil.copy(file.with_suffix(".txt"), new_path)


def split_train_val(args):
    all_file_list = sorted(Path(args.dataset_dir).glob("**/*.[jJpP][pPnN][gG]"))

    train_file_list = []
    val_file_list = []
    test_file_list = []

    train_file_count = int(len(all_file_list) * float(args.train_percentage / 10))
    val_file_count = int(len(all_file_list) * float(args.val_percentage / 10))
    test_file_count = int(len(all_file_list) * float(args.test_percentage / 10))

    for _ in range(train_file_count):
        train_file_list.append(all_file_list.pop(randrange(len(all_file_list))))
    for _ in range(val_file_count):
        val_file_list.append(all_file_list.pop(randrange(len(all_file_list))))
    for _ in range(test_file_count):
        test_file_list.append(all_file_list.pop(randrange(len(all_file_list))))

    create_dirs_and_move_files(train_file_list, "./train")
    create_dirs_and_move_files(val_file_list, "./val")
    create_dirs_and_move_files(test_file_list, "./test")

    print(
        f"train split: {len(train_file_list)}, val split: {len(val_file_list)}, test split: {len(test_file_list)}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument(
        "train_percentage", type=int, help="train split percentage (0-10)"
    )
    parser.add_argument("val_percentage", type=int)
    parser.add_argument("test_percentage", type=int)
    args = parser.parse_args()
    split_train_val(args)
