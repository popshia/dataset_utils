import argparse
import os
import shutil
from pathlib import Path

ORG_LABEL_DICT = {
    "car": "0",
    "taxi": "1",
    "pickuptruck": "2",
    "bus": "3",
    "truck": "4",
    "motor": "5",
    "bike": "6",
    "pedestrian": "7",
}

NEW_LABEL_DICT = {
    "small_car": "0",
    "large_car": "1",
    "motor": "2",
}


def tweak_label(args):
    all_txt_list = sorted(Path(args.input_dir).glob("**/*.txt"))

    for txt in all_txt_list:
        if not txt.match("classes.txt"):
            with open(txt.resolve().as_posix(), "r") as org_txt:
                output_dir = Path.cwd()
                parents = txt.parent.as_posix().split("/")[1:]
                parents.insert(0, args.output_dir)

                for parent in parents:
                    output_dir = output_dir.joinpath(parent)

                Path.mkdir(output_dir, parents=True, exist_ok=True)
                new_txt = output_dir.as_posix() + "/" + txt.name
                print(txt, new_txt)
                print("-" * os.get_terminal_size().columns)

                labels = org_txt.readlines()
                with open(new_txt, "w") as new_txt:
                    for label in labels:
                        org_label = label.split(" ")[0]
                        if org_label == ORG_LABEL_DICT["car"]:
                            new_txt.write(
                                label.replace(
                                    ORG_LABEL_DICT["car"],
                                    NEW_LABEL_DICT["small_car"],
                                    1,
                                )
                            )
                        elif org_label == ORG_LABEL_DICT["taxi"]:
                            new_txt.write(
                                label.replace(
                                    ORG_LABEL_DICT["taxi"],
                                    NEW_LABEL_DICT["small_car"],
                                    1,
                                )
                            )
                        elif org_label == ORG_LABEL_DICT["pickuptruck"]:
                            new_txt.write(
                                label.replace(
                                    ORG_LABEL_DICT["pickuptruck"],
                                    NEW_LABEL_DICT["small_car"],
                                    1,
                                )
                            )
                        elif org_label == ORG_LABEL_DICT["bus"]:
                            new_txt.write(
                                label.replace(
                                    ORG_LABEL_DICT["bus"],
                                    NEW_LABEL_DICT["large_car"],
                                    1,
                                )
                            )
                        elif org_label == ORG_LABEL_DICT["truck"]:
                            new_txt.write(
                                label.replace(
                                    ORG_LABEL_DICT["truck"],
                                    NEW_LABEL_DICT["large_car"],
                                    1,
                                )
                            )
                        elif org_label == ORG_LABEL_DICT["motor"]:
                            new_txt.write(
                                label.replace(
                                    ORG_LABEL_DICT["motor"],
                                    NEW_LABEL_DICT["motor"],
                                    1,
                                )
                            )
                        else:
                            continue

            if txt.with_suffix(".png").exists():
                shutil.copy(txt.with_suffix(".png").resolve(), output_dir)
            elif txt.with_suffix(".jpg").exists():
                shutil.copy(txt.with_suffix(".jpg").resolve(), output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    tweak_label(args)
