#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import xml.etree.ElementTree as ET
from typing import Tuple

def _indent(elem, level=0):
    """
    為了讓輸出 XML 比較好讀：遞迴縮排（兼容舊版 Python，避免依賴 ET.indent）
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            _indent(e, level + 1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i

def process_annotation_tree(tree: ET.ElementTree, src_class: str, dst_class: str = None) -> Tuple[int, int, int]:
    """
    處理一個 VOC XML 的 ElementTree：
    - 僅提供 src_class：刪除所有 <object><name>==src_class 的物件節點
    - 同時提供 dst_class：將所有 <object><name>==src_class 的名稱改成 dst_class

    回傳 (changed_cnt, removed_cnt, total_objects_before)
    """
    root = tree.getroot()
    objects = root.findall('object')
    total = len(objects)
    changed = 0
    removed = 0

    # 為了安全遍歷，用 list() 固定快照
    for obj in list(objects):
        name_el = obj.find('name')
        if name_el is None:
            continue
        if name_el.text == src_class:
            if dst_class is None:
                root.remove(obj)
                removed += 1
            else:
                name_el.text = dst_class
                changed += 1

    return changed, removed, total

def process_single_file(in_path: str, src_class: str, dst_class: str = None,
                        out_path: str = None, dry_run: bool = False) -> Tuple[int, int, int, str]:
    """
    處理單一 XML 檔案。若 out_path 為 None 則覆蓋原檔。
    回傳 (changed_cnt, removed_cnt, total_before, out_path_used)
    """
    try:
        tree = ET.parse(in_path)
    except ET.ParseError as e:
        raise RuntimeError(f"XML 格式錯誤：{in_path} ({e})")

    changed, removed, total = process_annotation_tree(tree, src_class, dst_class)

    # 只有真的有變動才輸出
    out_path_used = out_path or in_path
    if not dry_run and (changed > 0 or removed > 0):
        _indent(tree.getroot())
        tree.write(out_path_used, encoding='utf-8', xml_declaration=True)

    return changed, removed, total, out_path_used

def batch_process_folder(folder_path: str, src_class: str, dst_class: str = None,
                         ext: str = '.xml', output_folder: str = None, dry_run: bool = False) -> None:
    """
    遞迴掃描資料夾，處理所有指定副檔名的 VOC XML 標註檔：
    - 若提供 output_folder：將結果寫入對應相對路徑（不覆蓋原始檔）
    - 否則直接覆蓋原檔
    - 加上 --dry-run 可先預覽變更（不寫檔）
    """
    handled = 0
    changed_files = 0
    total_changed = 0
    total_removed = 0

    for root, _, files in os.walk(folder_path):
        for fname in files:
            if not fname.lower().endswith(ext.lower()):
                continue
            in_path = os.path.join(root, fname)

            # 決定輸出路徑
            if output_folder:
                rel_dir = os.path.relpath(root, folder_path)
                out_dir = os.path.join(output_folder, rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, fname)
            else:
                out_path = None  # 覆蓋

            try:
                changed, removed, total, out_used = process_single_file(
                    in_path, src_class, dst_class, out_path, dry_run=dry_run
                )
            except Exception as e:
                print(f"[錯誤] {in_path}: {e}")
                continue

            handled += 1
            if changed or removed:
                changed_files += 1
                total_changed += changed
                total_removed += removed
                action = f"改名 {src_class} → {dst_class}" if dst_class else f"刪除 {src_class}"
                flag = "(dry-run)" if dry_run else ""
                print(f"[{action}] {out_used}  變更: 改名 {changed}、刪除 {removed}、原有物件 {total} {flag}")
            else:
                print(f"[無變更] {in_path}")

    print("—"*60)
    print(f"已處理檔案：{handled}，有變更的檔案：{changed_files}")
    if dst_class:
        print(f"總改名數量：{total_changed}")
    print(f"總刪除數量：{total_removed}")
    print("全部處理完成。")

def main():
    parser = argparse.ArgumentParser(
        description="刪除或轉換 Pascal VOC XML 標註中的類別名稱，可批次處理資料夾。"
    )
    parser.add_argument('-f', '--folder', required=True, help="要處理的資料夾路徑")
    parser.add_argument('-c', '--class', dest='src_class', required=True, help="原本的類別名稱（如 'detect'）")
    parser.add_argument('-t', '--to-class', dest='dst_class', help="要轉成的類別名稱（不指定則刪除 src_class）")
    parser.add_argument('-e', '--ext', default='.xml', help="要處理的檔案副檔名，預設為 .xml")
    parser.add_argument('-o', '--output-folder', dest='output_folder',
                        help="輸出資料夾路徑，若指定則不覆蓋原始檔案")
    parser.add_argument('--dry-run', action='store_true', help="只預覽變更，不輸出檔案")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"錯誤：找不到資料夾 {args.folder}")
        return

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    batch_process_folder(
        folder_path=args.folder,
        src_class=args.src_class,
        dst_class=args.dst_class,
        ext=args.ext,
        output_folder=args.output_folder,
        dry_run=args.dry_run
    )

if __name__ == '__main__':
    main()
