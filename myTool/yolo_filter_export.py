#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import shutil
from typing import List, Optional, Set

IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
LABEL_EXT_DEFAULT = ".txt"

def has_any_class_in_label(label_path: str, class_ids: Set[str]) -> bool:
    """檢查 YOLO 標註檔是否包含指定任一類別（比對第一欄 class id）。"""
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if parts and parts[0] in class_ids:
                    return True
    except Exception as e:
        print(f"[警告] 無法讀取標註：{label_path}，錯誤：{e}")
    return False

def _replace_one_component(path: str, src_name_lc: str, dst_name: str) -> Optional[str]:
    """將路徑中的某一節（不分大小寫）由 src_name_lc 替換為 dst_name，回傳替換後路徑。找不到則回傳 None。"""
    parts = path.split(os.sep)
    for i, p in enumerate(parts):
        if p.lower() == src_name_lc:
            q = parts[:]
            q[i] = dst_name
            return os.sep.join(q)
    return None

def find_image_for_label(label_path: str, img_exts: List[str]) -> Optional[str]:
    """
    嘗試依據標註檔找到對應影像檔：
    1) 與標註同目錄、同檔名（不同副檔名）
    2) 若標註在 .../labels/...，嘗試將 labels 換成 images
    3) 若標註在 .../label/...，嘗試將 label 換成 images
    """
    stem = os.path.splitext(os.path.basename(label_path))[0]
    label_dir = os.path.dirname(label_path)

    # 1) 同目錄
    for ext in img_exts:
        cand = os.path.join(label_dir, stem + ext)
        if os.path.isfile(cand):
            return cand

    # 2) labels -> images
    cand_dir = _replace_one_component(label_dir, "labels", "images")
    if cand_dir:
        for ext in img_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    # 3) label -> images
    cand_dir = _replace_one_component(label_dir, "label", "images")
    if cand_dir:
        for ext in img_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    return None

def copy_file(src: str, dst: str):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

def scan_and_export(
    dataset_root: str,
    class_ids: List[str],
    output_root: str,
    label_ext: str = LABEL_EXT_DEFAULT,
    img_exts: Optional[List[str]] = None,
    dry_run: bool = False,
    strict_pair: bool = True,
):
    """
    遞迴掃描 dataset_root，找出含有任一 `class_ids` 的標註檔，並把該標註與對應影像複製到 output_root。
    - 保留相對於 dataset_root 的路徑結構。
    - strict_pair=True 時：找不到影像就跳過（只警告不複製）。
      strict_pair=False 時：即使找不到影像，仍會複製標註檔。
    """
    if img_exts is None:
        img_exts = IMG_EXTS_DEFAULT

    class_id_set: Set[str] = set(class_ids)

    total_labels = 0
    matched = 0
    copied_pairs = 0
    copied_labels_only = 0
    missing_images = 0

    for root, _, files in os.walk(dataset_root):
        for fname in files:
            if not fname.lower().endswith(label_ext.lower()):
                continue

            total_labels += 1
            label_path = os.path.join(root, fname)

            if not has_any_class_in_label(label_path, class_id_set):
                continue

            matched += 1

            # 找對應影像
            img_path = find_image_for_label(label_path, img_exts)

            # 決定輸出路徑（保留 dataset_root 之下的相對路徑）
            rel_label_path = os.path.relpath(label_path, dataset_root)
            out_label_path = os.path.join(output_root, rel_label_path)

            if img_path:
                rel_img_path = os.path.relpath(img_path, dataset_root)
                out_img_path = os.path.join(output_root, rel_img_path)

                if dry_run:
                    print(f"[DRY-RUN] 複製影像：{img_path} -> {out_img_path}")
                    print(f"[DRY-RUN] 複製標註：{label_path} -> {out_label_path}")
                else:
                    copy_file(img_path, out_img_path)
                    copy_file(label_path, out_label_path)
                copied_pairs += 1
            else:
                missing_images += 1
                msg = f"[警告] 找不到影像，已{'略過' if strict_pair else '僅複製標註'}：{label_path}"
                print(msg)
                if not strict_pair:
                    if dry_run:
                        print(f"[DRY-RUN] 複製標註：{label_path} -> {out_label_path}")
                    else:
                        copy_file(label_path, out_label_path)
                    copied_labels_only += 1

    class_list_str = ",".join(sorted(class_id_set))
    print("\n=== 統計 ===")
    print(f"掃描標註檔數：{total_labels}")
    print(f"符合類別 {class_list_str} 的標註：{matched}")
    print(f"成功複製（影像+標註）對數：{copied_pairs}")
    print(f"僅複製標註數（缺影像）：{copied_labels_only}")
    print(f"缺少影像數：{missing_images}")
    print(f"輸出目錄：{output_root}")

def _parse_class_ids(tokens: List[str]) -> List[str]:
    """
    將使用者輸入的 class id token（可能含逗號或多個值）攤平成字串清單。
    例如：["0,3", "5"] -> ["0","3","5"]
    """
    out: List[str] = []
    for t in tokens:
        for x in t.split(","):
            x = x.strip()
            if x != "":
                out.append(x)
    return out

def main():
    parser = argparse.ArgumentParser(
        description="篩出含指定類別（可多個）的 YOLO 影像與標註，並另存到指定目錄（保留相對路徑結構）。"
    )
    parser.add_argument("-s", "--src", required=True, help="資料集根目錄（會被遞迴掃描）")
    # 支援 -c / --class-id / --class-ids，多值或逗號分隔
    parser.add_argument(
        "-c", "--class-id", "--class-ids",
        dest="class_ids",
        nargs="+",
        required=True,
        help="要篩選的類別編號，可多個。例：-c 0 3 5 或 -c 0,3,5"
    )
    parser.add_argument("-o", "--out", required=True, help="輸出根目錄")
    parser.add_argument("-e", "--label-ext", default=LABEL_EXT_DEFAULT, help="標註副檔名，預設 .txt")
    parser.add_argument(
        "--img-exts",
        default=",".join(IMG_EXTS_DEFAULT),
        help="影像副檔名清單（以逗號分隔），預設：.jpg,.jpeg,.png,.bmp,.tif,.tiff",
    )
    parser.add_argument("--dry-run", action="store_true", help="僅列印將要複製的檔案，不實際動作")
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="若找不到對應影像，仍複製標註檔（預設嚴格模式：跳過）。",
    )

    args = parser.parse_args()

    dataset_root = os.path.abspath(args.src)
    output_root = os.path.abspath(args.out)
    class_ids = _parse_class_ids(args.class_ids)
    label_ext = args.label_ext
    img_exts = [x.strip() if x.strip().startswith(".") else f".{x.strip()}" for x in args.img_exts.split(",")]

    os.makedirs(output_root, exist_ok=True)

    scan_and_export(
        dataset_root=dataset_root,
        class_ids=class_ids,
        output_root=output_root,
        label_ext=label_ext,
        img_exts=img_exts,
        dry_run=args.dry_run,
        strict_pair=not args.non_strict,
    )

if __name__ == "__main__":
    main()
