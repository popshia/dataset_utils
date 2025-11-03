#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_empty_labels.py

給定資料夾根目錄，遍歷底下所有檔案與資料夾；
若發現某個「標註文字檔」為空（只含空白/換行皆視為空），
就刪除同資料夾中以該標註檔名為前綴(或同名)的相關檔案
（如 .png, .jpg, .txt 等）。

預設為「試跑」模式（不真的刪除），加上 --yes 才會實際刪除。

使用方式：
    python clean_empty_labels.py <ROOT_DIR> [--yes] [--mode {prefix,same-stem}] 
                                 [--ext .png .jpg .jpeg .txt .xml ...] [--follow-symlinks]

參數說明：
- ROOT_DIR：要掃描的資料夾根目錄。
- --yes：真的執行刪除（沒有此旗標則只會列出要刪除的目標）。
- --mode：
    prefix     -> 刪除同資料夾底下「檔名以 labelfile 為前綴」的檔案（預設）。
    same-stem  -> 只刪除「檔名主幹(stem)相同」的檔案（較保守）。
- --ext：指定要刪除的副檔名集合（預設為常見影像/標註副檔名）。
          若想刪除所有前綴相符的檔案，請改用 --all-extensions。
- --all-extensions：不檢查副檔名，凡是符合前綴/同名規則的檔案都刪除（請小心使用）。
- --follow-symlinks：在走訪目錄時跟隨符號連結。

範例：
    # 僅預覽會刪除哪些檔案
    python clean_empty_labels.py /path/to/dataset

    # 真的刪除，前綴匹配
    python clean_empty_labels.py /path/to/dataset --yes --mode prefix

    # 真的刪除，僅刪除同 stem 的檔案（較保守）
    python clean_empty_labels.py /path/to/dataset --yes --mode same-stem

    # 真的刪除，指定要刪的副檔名集合
    python clean_empty_labels.py /path/to/dataset --yes --ext .png .jpg .jpeg .txt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Set


DEFAULT_EXTS: Set[str] = {
    # 影像
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp",
    # 視訊快照/其他常見格式
    ".ppm", ".pgm",
    # 標註 / 伴隨檔
    ".txt", ".json", ".xml", ".csv", ".yml", ".yaml",
    ".npz", ".npy",
    ".label", ".labels",
}

TEXT_EXTS: Set[str] = {".txt", ".json", ".xml", ".yml", ".yaml"}


def read_text_safely(p: Path) -> str:
    """
    嘗試以 utf-8 讀取，不行則改用 latin-1；忽略無法解碼字元。
    僅在副檔名屬於 TEXT_EXTS 時才讀取。
    """
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return p.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""


def is_label_file_empty(label_path: Path) -> bool:
    """
    判斷標註檔是否「空白或僅空白字元/換行」。
    （常見 YOLO 空標註為 0 行）
    """
    try:
        if label_path.stat().st_size == 0:
            return True
    except FileNotFoundError:
        return False

    # 僅在是純文字副檔名時進一步檢查
    if label_path.suffix.lower() in TEXT_EXTS:
        content = read_text_safely(label_path)
        if content.strip() == "":
            return True
    return False


def gather_targets_to_delete(
    directory: Path,
    label_file: Path,
    mode: str,
    exts: Set[str],
    all_extensions: bool
) -> List[Path]:
    """
    在 label_file 所在的資料夾中，找出要刪除的檔案清單。
    - mode == "prefix": 刪除檔名以 label 檔名主幹為前綴的所有檔案。
    - mode == "same-stem": 只刪除檔名主幹相同的檔案。
    - 若 all_extensions 為 True，則不檢查副檔名；否則僅刪 exts 中的副檔名。
    """
    stem = label_file.stem
    results: List[Path] = []
    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p == label_file:
            # 標註檔本身當然也要包含在刪除清單內
            match = True
        else:
            if mode == "prefix":
                match = p.name.startswith(stem)
            else:  # same-stem
                match = p.stem == stem

        if not match:
            continue

        if all_extensions or p.suffix.lower() in exts:
            results.append(p)

    return results


def iter_all_files(root: Path, follow_symlinks: bool) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        d = Path(dirpath)
        for fn in filenames:
            yield d / fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="掃描資料夾，若某 *label*.txt 為空，刪除同前綴/同名的所有關聯檔案。"
    )
    parser.add_argument("root", type=str, help="資料夾根目錄路徑")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="真的執行刪除（預設只會列出要刪除的項目）",
    )
    parser.add_argument(
        "--mode",
        choices=["prefix", "same-stem"],
        default="prefix",
        help="刪除模式：prefix=以標註檔名為前綴的檔案；same-stem=檔名主幹完全相同（較保守）。預設 prefix。",
    )
    parser.add_argument(
        "--ext",
        nargs="+",
        default=None,
        help="要刪除的副檔名（包含小數點），例如：--ext .png .jpg .txt。若未指定則使用內建常見清單。",
    )
    parser.add_argument(
        "--all-extensions",
        action="store_true",
        help="不限制副檔名，只要前綴/同名符合就刪除（高風險，請小心）。",
    )
    parser.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="走訪目錄時跟隨符號連結（預設不跟隨）。",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"[錯誤] 路徑不存在或不是資料夾：{root}")
        raise SystemExit(1)

    exts: Set[str] = set(e.lower() for e in (args.ext or DEFAULT_EXTS))
    all_extensions: bool = bool(args.all_extensions)

    total_labels_checked = 0
    total_empty_labels = 0
    total_files_to_delete = 0
    total_deleted = 0

    print(f"[資訊] 掃描根目錄：{root}")
    print(f"[資訊] 模式：{args.mode}，副檔名限制：{'不限(ALL)' if all_extensions else ', '.join(sorted(exts))}")
    print(f"[資訊] 實際刪除：{'是' if args.yes else '否(試跑)'}")
    print("-" * 80)

    for p in iter_all_files(root, args.follow_symlinks):
        # 只把 .txt 視為「標註文字檔」的觸發條件；
        # 若想用別的標註格式，可在此加入判斷。
        if p.suffix.lower() != ".txt":
            continue

        total_labels_checked += 1
        if not is_label_file_empty(p):
            continue

        total_empty_labels += 1
        targets = gather_targets_to_delete(p.parent, p, args.mode, exts, all_extensions)

        if not targets:
            continue

        print(f"[標註空白] {p}")
        for t in sorted(targets):
            total_files_to_delete += 1
            print(f"  -> 目標：{t}")
            if args.yes:
                try:
                    t.unlink(missing_ok=True)
                    total_deleted += 1
                except Exception as e:
                    print(f"     [刪除失敗] {t}，原因：{e}")

    print("-" * 80)
    print(f"[統計] 檢查到的 .txt 標註檔數量：{total_labels_checked}")
    print(f"[統計] 其中空白標註檔數量：{total_empty_labels}")
    print(f"[統計] 擬刪除檔案總數：{total_files_to_delete}")
    if args.yes:
        print(f"[統計] 成功刪除：{total_deleted}")
    else:
        print("[提示] 目前為試跑模式，如需真的刪除請加入 --yes")
    print("[完成]")
    

if __name__ == "__main__":
    main()
