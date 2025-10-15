from __future__ import annotations

"""
Split‑Dataset Optimized v1.1
---------------------------
* 新增 **--ignore-dirs** 參數（互動模式亦可輸入），可指定額外要**排除掃描**的子資料夾或關鍵字
* **--keep-split-dirs** 可保留既有 train/val/test 目錄，若欲納入舊資料夾一起重新隨機分配可使用此旗標
* 其餘功能維持 v1.0：僅輸出影像路徑、支援搬移 / 只產生清單、Pathlib 改寫
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Set

# ---------------------------- 公用設定 ----------------------------
IMG_EXT: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
LABEL_EXT: Set[str] = {".txt",".json"}  # 依需求擴充
DEFAULT_SPLIT_DIRS: Set[str] = {"train", "val", "test"}
LIST_FILES = {"train.txt", "val.txt", "test.txt"}

# ---------------------------- 工具函式 ----------------------------

def list_data_files(root: Path, ignore_dirs: Set[str]) -> List[Path]:
    """遞迴列出 root 下所有影像/標註檔 (回傳 **相對** 路徑)。"""
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            # 如遇需忽略的資料夾名稱 (完全相符或包含關鍵字) → skip 整個子樹
            if any(kw.lower() in p.name.lower() for kw in ignore_dirs):
                # rglob 沒有直接 skip 功能；這樣就不會再進入下層
                continue
            continue
        if p.name in LIST_FILES:
            continue
        if p.suffix.lower() in IMG_EXT | LABEL_EXT:
            files.append(p.relative_to(root))
    return files


def group_by_stem(paths: Iterable[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for rel in paths:
        key = rel.with_suffix("")  # stem 包含子層目錄
        groups.setdefault(str(key), []).append(rel)
    return groups


def find_image(group: Sequence[Path]) -> Path | None:
    for p in group:
        if p.suffix.lower() in IMG_EXT:
            return p
    return None


def move_group(group: Sequence[Path], src_root: Path, dst_root: Path) -> Path | None:
    img_rel = find_image(group)
    for rel in group:
        src = src_root / rel
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, dst)
    if img_rel is None:
        return None
    return (dst_root / img_rel).resolve()

# ---------------------------- 主流程 ----------------------------

def split_files(
    folder_path: str | Path,
    ratio: Tuple[int, int, int] = (8, 1, 1),
    reorganize: bool = True,
    ignore_dirs: Set[str] | None = None,
    keep_default_split: bool = True,
) -> None:
    root = Path(folder_path).expanduser().resolve()
    if not root.is_dir():
        print(f"[錯誤] 指定路徑不存在：{root}")
        return

    ignore_set: Set[str] = set()
    if keep_default_split:
        ignore_set.update(DEFAULT_SPLIT_DIRS)
    if ignore_dirs:
        ignore_set.update(ignore_dirs)

    all_files = list_data_files(root, ignore_set)
    if not all_files:
        print("[警告] 找不到任何符合條件的檔案。")
        return

    # 分群、打亂、切分
    groups = list(group_by_stem(all_files).values())
    random.shuffle(groups)
    s = sum(ratio)
    n_total = len(groups)
    n_train = int(n_total * ratio[0] / s)
    n_val = int(n_total * ratio[1] / s)

    train_groups = groups[: n_train]
    val_groups = groups[n_train : n_train + n_val]
    test_groups = groups[n_train + n_val :]

    def process(groups: Sequence[Sequence[Path]], dst: Path | None) -> List[str]:
        collected: List[str] = []
        for grp in groups:
            if reorganize and dst is not None:
                img_abs = move_group(grp, root, dst)
            else:
                img_rel = find_image(grp)
                img_abs = (root / img_rel).resolve() if img_rel else None
            if img_abs is not None:
                collected.append(str(img_abs))
        return collected

    # 決定目的資料夾
    if reorganize:
        train_dir, val_dir, test_dir = (root / "train", root / "val", root / "test")
        for d in (train_dir, val_dir, test_dir):
            d.mkdir(parents=True, exist_ok=True)
    else:
        train_dir = val_dir = test_dir = None  # type: ignore

    lists = {
        "train": process(train_groups, train_dir),
        "val": process(val_groups, val_dir),
        "test": process(test_groups, test_dir),
    }

    # 輸出清單檔
    for name, paths in lists.items():
        txt_file = root / f"{name}.txt"
        txt_file.write_text("\n".join(paths), encoding="utf-8")
        print(f"✔ 已產生 {txt_file} ({len(paths)})")

    print("=== 分割完成 ===")

# ---------------------------- CLI 與互動 ----------------------------

if __name__ == "__main__":
    interactive = sys.stdin.isatty()

    def parse_ratio(r: str) -> Tuple[int, int, int]:
        parts = [int(x) for x in r.replace(",", ":").split(":")]
        if len(parts) != 3 or any(x < 0 for x in parts):
            raise ValueError
        return tuple(parts)  # type: ignore

    if interactive:
        folder = input("請輸入資料夾路徑：").strip().strip("'\"")
        ans = input("是否將檔案搬移到 train/val/test？(Y/n) [預設 Y]：").strip().lower()
        reorganize = ans != "n"
        ratio_str = input("請輸入 train:val:test 比例 (預設 8:1:1)：").strip() or "8:1:1"
        try:
            ratio = parse_ratio(ratio_str)
        except ValueError:
            print("比例格式錯誤，使用預設 (8,1,1)")
            ratio = (8, 1, 1)
        # 讓用戶自行輸入忽略資料夾關鍵字
        ignore_str = input("要忽略哪些子資料夾關鍵字？(逗號分隔，可留空)：").strip()
        ignore_set = {s.strip() for s in ignore_str.split(",") if s.strip()} if ignore_str else set()
        keep_split = input("是否保留既有 train/val/test？(Y/n) [預設 Y]：").strip().lower() != "n"
        split_files(folder, ratio, reorganize, ignore_set, keep_split)
    else:
        parser = argparse.ArgumentParser(description="隨機將資料集分成 train/val/test")
        parser.add_argument("folder", help="資料集根目錄")
        parser.add_argument("--ratio", default="8:1:1", help="train:val:test 比例")
        parser.add_argument("--no-reorganize", action="store_true", help="僅產生 txt，不搬移檔案")
        parser.add_argument(
            "--ignore-dirs",
            default="",
            help="額外忽略的子資料夾名稱/關鍵字 (以逗號分隔)",
        )
        parser.add_argument(
            "--keep-split-dirs",
            action="store_true",
            help="保留既有 train/val/test 目錄 (將它們也納入掃描)",
        )
        args = parser.parse_args()
        try:
            ratio_tuple = parse_ratio(args.ratio)
        except ValueError:
            print("[警告] 比例格式錯誤，使用預設 (8,1,1)")
            ratio_tuple = (8, 1, 1)
        ignore_set = {s.strip() for s in args.ignore_dirs.split(",") if s.strip()}
        split_files(
            args.folder,
            ratio_tuple,
            not args.no_reorganize,
            ignore_set,
            not args.keep_split_dirs,
        )
