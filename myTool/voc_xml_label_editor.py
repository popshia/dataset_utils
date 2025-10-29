#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import xml.etree.ElementTree as ET
from typing import Tuple, Optional, Set, List

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

def process_annotation_tree(
    tree: ET.ElementTree,
    src_class: Optional[str] = None,
    dst_class: Optional[str] = None,
    keep_classes: Optional[Set[str]] = None
) -> Tuple[int, int, int]:
    """
    處理一個 VOC XML 的 ElementTree：
    - (改名) 若同時提供 src_class 與 dst_class：將所有 <object><name>==src_class 改成 dst_class
    - (刪除) 若只提供 src_class：刪除所有 <object><name>==src_class 的物件節點
    - (保留) 若提供 keep_classes（可複數）：僅保留名稱在 keep_classes 內的物件，其餘刪除
      ※ 若同時有改名與保留：會先執行改名，再執行保留，以避免改名目標被誤刪

    回傳 (changed_cnt, removed_cnt, total_objects_before)
    """
    root = tree.getroot()
    objects = root.findall('object')
    total = len(objects)
    changed = 0
    removed = 0

    # 1) 改名 / 刪除 src_class
    if src_class:
        for obj in list(objects):  # 固定快照
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

    # 2) 僅保留 keep_classes（若有）
    if keep_classes:
        # 重新取得（可能已經改名/刪除過）
        for obj in list(root.findall('object')):
            name_el = obj.find('name')
            if name_el is None:
                continue
            if name_el.text not in keep_classes:
                root.remove(obj)
                removed += 1

    return changed, removed, total

def process_single_file(in_path: str,
                        src_class: Optional[str] = None,
                        dst_class: Optional[str] = None,
                        keep_classes: Optional[Set[str]] = None,
                        out_path: Optional[str] = None,
                        dry_run: bool = False) -> Tuple[int, int, int, str]:
    """
    處理單一 XML 檔案。若 out_path 為 None 則覆蓋原檔。
    回傳 (changed_cnt, removed_cnt, total_before, out_path_used)
    """
    try:
        tree = ET.parse(in_path)
    except ET.ParseError as e:
        raise RuntimeError(f"XML 格式錯誤：{in_path} ({e})")

    changed, removed, total = process_annotation_tree(
        tree, src_class=src_class, dst_class=dst_class, keep_classes=keep_classes
    )

    # 只有真的有變動才輸出
    out_path_used = out_path or in_path
    if not dry_run and (changed > 0 or removed > 0):
        _indent(tree.getroot())
        tree.write(out_path_used, encoding='utf-8', xml_declaration=True)

    return changed, removed, total, out_path_used

def batch_process_folder(folder_path: str,
                         src_class: Optional[str] = None,
                         dst_class: Optional[str] = None,
                         keep_classes: Optional[Set[str]] = None,
                         ext: str = '.xml',
                         output_folder: Optional[str] = None,
                         dry_run: bool = False) -> None:
    """
    遞迴掃描資料夾，處理所有指定副檔名的 VOC XML 標註檔：
    - 若提供 output_folder：將結果寫入對應相對路徑（不覆蓋原始檔）
    - 否則直接覆蓋原檔
    - 加上 --dry-run 可先預覽變更（不寫檔）
    - keep_classes（可複數）可用來僅保留特定類別（其他全部刪除）
    """
    handled = 0
    changed_files = 0
    total_changed = 0
    total_removed = 0

    for root_dir, _, files in os.walk(folder_path):
        for fname in files:
            if not fname.lower().endswith(ext.lower()):
                continue
            in_path = os.path.join(root_dir, fname)

            # 決定輸出路徑
            if output_folder:
                rel_dir = os.path.relpath(root_dir, folder_path)
                out_dir = os.path.join(output_folder, rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, fname)
            else:
                out_path = None  # 覆蓋

            try:
                changed, removed, total, out_used = process_single_file(
                    in_path,
                    src_class=src_class,
                    dst_class=dst_class,
                    keep_classes=keep_classes,
                    out_path=out_path,
                    dry_run=dry_run
                )
            except Exception as e:
                print(f"[錯誤] {in_path}: {e}")
                continue

            handled += 1
            if changed or removed:
                changed_files += 1
                total_changed += changed
                total_removed += removed

                actions: List[str] = []
                if src_class and dst_class:
                    actions.append(f"改名 {src_class} → {dst_class}")
                elif src_class and not dst_class:
                    actions.append(f"刪除 {src_class}")
                if keep_classes:
                    actions.append(f"保留 {sorted(keep_classes)}（其餘刪除）")

                action = " & ".join(actions) if actions else "變更"
                flag = "(dry-run)" if dry_run else ""
                print(f"[{action}] {out_used}  變更: 改名 {changed}、刪除 {removed}、原有物件 {total} {flag}")
            else:
                print(f"[無變更] {in_path}")

    print("—"*60)
    print(f"已處理檔案：{handled}，有變更的檔案：{changed_files}")
    if src_class and dst_class:
        print(f"總改名數量：{total_changed}")
    print(f"總刪除數量：{total_removed}")
    print("全部處理完成。")

def _parse_keep_list(raw_list: Optional[List[str]]) -> Optional[Set[str]]:
    """
    支援：
    - 多次 -k/--keep-classes 重複提供
    - 逗號分隔的清單（會自動拆解）
    - 自動過濾空字串
    """
    if not raw_list:
        return None
    acc: List[str] = []
    for token in raw_list:
        acc.extend([t.strip() for t in token.split(',') if t.strip()])
    return set(acc) if acc else None

def main():
    parser = argparse.ArgumentParser(
        description="刪除/改名/保留 Pascal VOC XML 標註中的類別，可批次處理資料夾。"
    )
    parser.add_argument('-f', '--folder', required=True, help="要處理的資料夾路徑")

    # 原本的刪除/改名邏輯
    parser.add_argument('-c', '--class', dest='src_class', help="原本的類別名稱（如 'detect'）")
    parser.add_argument('-t', '--to-class', dest='dst_class', help="要轉成的類別名稱（不指定則刪除 src_class）")

    # 新增：保留清單（可複數）
    parser.add_argument('-k', '--keep-classes', dest='keep_classes', nargs='+',
                        help="要『保留』的類別（可複數/可逗號分隔）。若指定，會刪除所有不在此清單內的物件。")

    parser.add_argument('-e', '--ext', default='.xml', help="要處理的檔案副檔名，預設為 .xml")
    parser.add_argument('-o', '--output-folder', dest='output_folder',
                        help="輸出資料夾路徑，若指定則不覆蓋原始檔案")
    parser.add_argument('--dry-run', action='store_true', help="只預覽變更，不輸出檔案")
    args = parser.parse_args()

    # 基本檢查
    if not os.path.isdir(args.folder):
        print(f"錯誤：找不到資料夾 {args.folder}")
        return

    if args.output_folder:
        os.makedirs(args.output_folder, exist_ok=True)

    # 至少需要有一種操作：src_class（刪除/改名）或 keep_classes（保留）
    keep_set = _parse_keep_list(args.keep_classes)
    if not args.src_class and not keep_set:
        print("錯誤：請至少指定一種操作：\n"
              "  -c/--class（搭配可選的 -t/--to-class），或\n"
              "  -k/--keep-classes（可多個）")
        return

    batch_process_folder(
        folder_path=args.folder,
        src_class=args.src_class,
        dst_class=args.dst_class,
        keep_classes=keep_set,
        ext=args.ext,
        output_folder=args.output_folder,
        dry_run=args.dry_run
    )

if __name__ == '__main__':
    main()
