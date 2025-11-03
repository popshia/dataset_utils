#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from typing import Iterable, List, Tuple

def normalize_exts(exts: Iterable[str]) -> Tuple[str, ...]:
    """將副檔名正規化為以點起頭的小寫形式（例：'.jpg'）。"""
    norm = []
    for e in exts:
        e = e.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = "." + e
        norm.append(e)
    return tuple(sorted(set(norm)))

def find_unannotated_images(
    directory: str,
    image_extensions: Tuple[str, ...],
    annotation_extensions: Tuple[str, ...],
    recursive: bool = False,
) -> List[str]:
    """
    找出缺少對應錨定檔案的過濾檔清單（僅在相同資料夾比對同名不同副檔名）。
    """
    targets: List[str] = []
    walk_iter = os.walk(directory) if recursive else [(directory, [], os.listdir(directory))]

    for root, _dirs, files in walk_iter:
        # 建立當前資料夾的檔名集合以加速 exists 判斷
        # 但因為要比對「同名不同副檔名」是否存在，所以仍逐一檢查
        lower_files = {f.lower() for f in files}

        for fname in files:
            name, ext = os.path.splitext(fname)
            if ext.lower() in image_extensions:
                # 是否存在任一對應錨定
                found = False
                for anchor_ext in annotation_extensions:
                    candidate = (name + anchor_ext).lower()
                    if candidate in lower_files:
                        found = True
                        break
                if not found:
                    targets.append(os.path.join(root, fname))
    return targets

def delete_files(files: Iterable[str]) -> Tuple[int, List[Tuple[str, str]]]:
    """
    刪除檔案，回傳成功數量與失敗列表。
    """
    failed: List[Tuple[str, str]] = []
    count = 0
    for f in files:
        try:
            os.remove(f)
            count += 1
            print(f"已刪除：{f}")
        except OSError as e:
            failed.append((f, str(e)))
            print(f"刪除失敗 {f}: {e}")
    return count, failed

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="刪除沒有對應錨定檔（.xml 或 .txt…）的過濾檔（.jpg/.png…）。"
    )
    parser.add_argument(
        "directory",
        help="要處理的目錄路徑",
    )
    parser.add_argument(
        "-a",
        "--anchor-ext",
        nargs="+",
        default=[".xml", ".txt"],
        help="錨定副檔名（可多個）。可填 '.xml' 或 'xml' 皆可，預設：.xml .txt",
    )
    parser.add_argument(
        "-f",
        "--filter-ext",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="過濾副檔名（可多個）。可填 '.jpg' 或 'jpg' 皆可，預設：.jpg .jpeg .png",
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="遞迴處理子資料夾",
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="僅列出將被刪除的檔案，不實際刪除",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="直接刪除，跳過互動式確認",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="輸出更多過程資訊",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    directory = args.directory
    if not os.path.isdir(directory):
        print(f"錯誤：'{directory}' 不是一個有效的目錄。")
        return

    image_extensions = normalize_exts(args.filter_ext)
    annotation_extensions = normalize_exts(args.anchor_ext)

    if args.verbose:
        print(f"目錄：{directory}")
        print(f"過濾副檔名：{', '.join(image_extensions)}")
        print(f"錨定副檔名：{', '.join(annotation_extensions)}")
        print(f"遞迴：{args.recursive}, 乾跑：{args.dry_run}, 自動確認：{args.yes}")

    files_to_delete = find_unannotated_images(
        directory=directory,
        image_extensions=image_extensions,
        annotation_extensions=annotation_extensions,
        recursive=args.recursive,
    )

    if not files_to_delete:
        print("沒有發現需要刪除的過濾檔案。")
        return

    print("將會刪除以下沒有對應錨定檔案的過濾：")
    for f in files_to_delete:
        print(f"- {f}")
    print(f"總計：{len(files_to_delete)} 檔")

    if args.dry_run:
        print("（乾跑模式）未執行刪除。")
        return

    if not args.yes:
        try:
            confirm = input("確定要刪除這些檔案嗎？(y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n操作已取消。")
            return
        if confirm != "y":
            print("操作已取消。")
            return

    deleted, failed = delete_files(files_to_delete)
    print(f"刪除完成。成功：{deleted}，失敗：{len(failed)}")
    if failed:
        print("以下檔案刪除失敗：")
        for f, msg in failed:
            print(f"- {f} 原因：{msg}")

if __name__ == "__main__":
    main()
