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
        print(new_path)
        print(file, file.with_suffix(".txt"))

        shutil.copy(file, new_path)
        if Path(file.with_suffix(".txt")).exists():
            shutil.copy(file.with_suffix(".txt"), new_path)

        print("-" * os.get_terminal_size().columns)


def split_train_val(args):
    all_file_list = sorted(Path(args.dataset_dir).glob("**/*.[jJpP][pPnN][gG]"))
    train_file_count = int(len(all_file_list) * float(args.train_percentage / 100))
    train_file_list = []
    for _ in range(train_file_count):
        train_file_list.append(all_file_list.pop(randrange(len(all_file_list))))
    val_file_list = all_file_list
    create_dirs_and_move_files(train_file_list, "./train")
    create_dirs_and_move_files(val_file_list, "./val")
    print(
        "train split: {}, val split: {}".format(
            len(train_file_list), len(val_file_list)
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir")
    parser.add_argument(
        "train_percentage", type=int, help="precentange of train split (1-100)"
    )
    args = parser.parse_args()
    split_train_val(args)
