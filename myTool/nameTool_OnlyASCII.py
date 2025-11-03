#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sanitize_filenames_interactive.py

遍歷指定資料夾中的所有檔案，預覽將移除的非 ASCII 字元後的新檔名，
讓使用者確認後再執行真正的更名操作。
"""
import os
import sys


def sanitize_str(s: str) -> str:
    """
    移除字串中所有非 ASCII 字元 (ord(c) >= 128)。
    若結果為空字串，回傳 '_' 以避免空檔名。
    """
    filtered = ''.join(c for c in s if ord(c) < 128)
    return filtered or '_'


def preview_changes(path: str) -> list:
    """
    掃描資料夾中所有檔案，產生 (old_name, proposed_new_name)
    的列表，但不實際執行更名。
    """
    changes = []
    for filename in os.listdir(path):
        full_old = os.path.join(path, filename)
        if not os.path.isfile(full_old):
            continue

        base, ext = os.path.splitext(filename)
        new_base = sanitize_str(base)
        new_ext  = sanitize_str(ext)
        new_name = new_base + new_ext

        if new_name != filename:
            # 尋找避免衝突的最終名稱
            counter = 1
            final_name = new_name
            full_new = os.path.join(path, final_name)
            while os.path.exists(full_new):
                final_name = f"{new_base}_{counter}" + new_ext
                full_new = os.path.join(path, final_name)
                counter += 1
            changes.append((filename, final_name))
    return changes


def apply_changes(path: str, changes: list):
    """
    執行更名操作，並列印每個檔案的更名過程。
    """
    for old_name, new_name in changes:
        full_old = os.path.join(path, old_name)
        full_new = os.path.join(path, new_name)
        os.rename(full_old, full_new)
        print(f"已重新命名: '{old_name}' → '{new_name}'")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("用法：python sanitize_filenames_interactive.py <資料夾路徑>")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"錯誤：找不到資料夾 {target_dir}")
        sys.exit(1)

    changes = preview_changes(target_dir)
    if not changes:
        print("所有檔案檔名皆已符合 ASCII 條件，無需更名。")
        sys.exit(0)

    print("以下為預覽的更名清單：")
    for old, new in changes:
        print(f"  {old}  →  {new}")

    confirm = input("是否要執行更名？[y/N]: ").strip().lower()
    if confirm == 'y' or confirm == 'yes':
        apply_changes(target_dir, changes)
        print("處理完成！")
    else:
        print("已取消更名。")
