import shutil
from pathlib import Path


def move_files_with_leading_0_or_1(src_dir: str, dest_dir: str):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)

    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    for txt_file in src_path.rglob("*.txt"):
        try:
            with txt_file.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.lstrip()  # ignore leading spaces
                    if stripped.startswith(("0", "1")):
                        shutil.move(str(txt_file), dest_path / txt_file.name)
                        print(f"Moved: {txt_file.name}, with class {stripped[0]}")
                        break  # stop checking once we know it matches
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")


if __name__ == "__main__":
    # Example usage:
    move_files_with_leading_0_or_1("./labels", "./other_classes")
