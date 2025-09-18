import re
from pathlib import Path

# Mapping of replacements
replacement_map = {"3": "0", "4": "1", "5": "2", "2": "3"}


def replace_leading_digit(line: str) -> str:
    """
    Replace the leading integer digit at the start of the line
    if it matches the replacement_map.
    """
    return re.sub(r"^([2-5])", lambda m: replacement_map[m.group(1)], line)


def process_txt_file(filepath: Path):
    """
    Read a text file, replace leading digits according to rules,
    and overwrite the file.
    """
    text = filepath.read_text(encoding="utf-8").splitlines(keepends=True)
    new_lines = [replace_leading_digit(line) for line in text]
    filepath.write_text("".join(new_lines), encoding="utf-8")


def process_directory(root_dir: Path):
    """
    Recursively find all .txt files in the directory and process them.
    """
    for filepath in root_dir.rglob("*.txt"):
        print(f"Processing {filepath}")
        process_txt_file(filepath)


if __name__ == "__main__":
    # Change this path to the directory you want to process
    target_directory = Path("./labels")
    process_directory(target_directory)
