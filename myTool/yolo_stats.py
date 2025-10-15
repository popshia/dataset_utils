#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import sys
from typing import Dict, List, Optional, Set, Tuple
try:
    # 讀取影像尺寸用（僅讀表頭，效能負擔小）
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # Pillow 未安裝時允許降級運作
try:
    # 若有安裝 tqdm，使用更美觀的進度列；否則退化為簡易百分比輸出到 STDERR
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None
IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
LABEL_EXT_DEFAULT = ".txt"


# --------- 進度列輔助 ---------
def _progress_iter(iterable, total: Optional[int], desc: str, enabled: bool):
    """
    包裝 iterable：
    - enabled 為 False 時直接回傳原 iterable。
    - 若可用 tqdm，使用 tqdm；否則以簡易百分比印到 STDERR。
    """
    if not enabled:
        for x in iterable:
            yield x
        return

    if _tqdm is not None:
        # 直接用 tqdm
        yield from _tqdm(iterable, total=total, desc=desc, unit="file")
    else:
        # 簡易百分比
        count = 0
        total = int(total) if total is not None else None
        for x in iterable:
            count += 1
            if total:
                pct = int(count * 100 / total)
                print(f"\r{desc}: {count}/{total} ({pct}%)", end="", file=sys.stderr)
            else:
                print(f"\r{desc}: {count}", end="", file=sys.stderr)
            yield x
        print("", file=sys.stderr)

# --------- 路徑配對輔助 ---------
def _replace_one_component(path: str, src_name_lc: str, dst_name: str) -> Optional[str]:
    parts = path.split(os.sep)
    for i, p in enumerate(parts):
        if p.lower() == src_name_lc:
            q = parts[:]
            q[i] = dst_name
            return os.sep.join(q)
    return None

def find_image_for_label(label_path: str, img_exts: List[str]) -> Optional[str]:
    """由標註推測影像檔路徑：同目錄；或 labels/label -> images。"""
    stem = os.path.splitext(os.path.basename(label_path))[0]
    label_dir = os.path.dirname(label_path)

    # 同目錄
    for ext in img_exts:
        cand = os.path.join(label_dir, stem + ext)
        if os.path.isfile(cand):
            return cand

    # labels -> images
    cand_dir = _replace_one_component(label_dir, "labels", "images")
    if cand_dir:
        for ext in img_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    # label -> images
    cand_dir = _replace_one_component(label_dir, "label", "images")
    if cand_dir:
        for ext in img_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    return None

def find_label_for_image(image_path: str, label_ext: str) -> Optional[str]:
    """由影像推測標註檔路徑：同目錄；或 images -> labels/label。"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.dirname(image_path)

    # 同目錄
    cand = os.path.join(img_dir, stem + label_ext)
    if os.path.isfile(cand):
        return cand

    # images -> labels
    cand_dir = _replace_one_component(img_dir, "images", "labels")
    if cand_dir:
        cand = os.path.join(cand_dir, stem + label_ext)
        if os.path.isfile(cand):
            return cand

    # images -> label
    cand_dir = _replace_one_component(img_dir, "images", "label")
    if cand_dir:
        cand = os.path.join(cand_dir, stem + label_ext)
        if os.path.isfile(cand):
            return cand

    return None

# --------- 標註解析 ---------
def parse_yolo_label_file(path: str) -> Tuple[int, Dict[int, int], Set[int], int]:
    """
    讀取 YOLO 標註檔：
    回傳 (有效行數, 各類別實例數, 本檔包含之類別集合, 無效行數)
    - YOLO 行格式至少需有：class_id x y w h（或更多；segmentation 也可，但第一欄必為 class_id）
    """
    class_counts: Dict[int, int] = defaultdict(int)
    classes_in_file: Set[int] = set()
    valid = 0
    invalid_lines = 0

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))  # 兼容 '0', '0.0'
                    class_counts[cls] += 1
                    classes_in_file.add(cls)
                    valid += 1
                except Exception:
                    invalid_lines += 1
    except Exception:
        # 無法讀檔一律視為 0 有效行、整檔無效
        invalid_lines += 1

    return valid, class_counts, classes_in_file, invalid_lines

# --------- 影像尺寸讀取 ---------
def get_image_size(path: str) -> Optional[Tuple[int, int]]:
    """回傳 (width, height)。若未安裝 Pillow 或讀取失敗則回傳 None。"""
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None

# --------- 統計資料結構 ---------
@dataclass
class YOLOStats:
    images_total: int = 0
    labels_total: int = 0
    labeled_images: int = 0
    unlabeled_images: int = 0
    labels_without_image: int = 0
    empty_label_files: int = 0
    invalid_label_lines: int = 0
    instances_total: int = 0
    instances_per_image_avg: float = 0.0
    instances_per_image_max: int = 0

    # 類別統計
    per_class_instances: Dict[int, int] = None
    per_class_image_count: Dict[int, int] = None
    image_size_counts: Dict[str, int] = None
# --------- 主流程 ---------
def scan_dataset(
    dataset_root: str,
    img_exts: List[str],
    label_ext: str,
    paired_only: bool = False,
    show_progress: bool = False,
) -> YOLOStats:
    stats = YOLOStats(
        per_class_instances=defaultdict(int),
        per_class_image_count=defaultdict(int),
        image_size_counts=defaultdict(int),
    )

    # 收集所有影像與標註檔
    images: List[str] = []
    labels: List[str] = []
    for root, _, files in os.walk(dataset_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            low = fname.lower()
            if any(low.endswith(ext.lower()) for ext in img_exts):
                images.append(fpath)
            elif low.endswith(label_ext.lower()):
                labels.append(fpath)

    stats.images_total = len(images)
    stats.labels_total = len(labels)

    # 單次走訪影像清單：同時統計「尺寸分布」與「有無對應標註」，並提供進度列
    labeled_images = 0
    for img in _progress_iter(images, total=len(images), desc="掃描影像（尺寸與標註配對）", enabled=show_progress):
        # 影像尺寸（若可用 Pillow）
        if Image is not None:
            sz = get_image_size(img)
            if sz:
                w, h = sz
                stats.image_size_counts[f"{w}x{h}"] += 1
        # 標註配對
        if find_label_for_image(img, label_ext):
            labeled_images += 1
    # 先算影像是否有標註
    labeled_images = 0
    for img in images:
        if find_label_for_image(img, label_ext):
            labeled_images += 1
    stats.labeled_images = labeled_images
    stats.unlabeled_images = stats.images_total - labeled_images

    # 計算標註是否有對應影像 & 類別統計
    instances_per_labeled_image: List[int] = []
    seen_class_in_image: Dict[int, int] = defaultdict(int)  # 暫存每個標註檔對應到之類別集合大小

    for lab in _progress_iter(labels, total=len(labels), desc="解析標註檔", enabled=show_progress):
        img = find_image_for_label(lab, img_exts)
        has_pair = img is not None

        if not has_pair:
            stats.labels_without_image += 1
            if paired_only:
                # 僅統計有配對者時，遇到孤兒標註就直接跳過解析
                continue

        valid, class_counts, cls_set, invalid_lines = parse_yolo_label_file(lab)
        stats.invalid_label_lines += invalid_lines
        if valid == 0:
            stats.empty_label_files += 1

        # 若設定 paired_only，只有在 has_pair 時才納入統計
        if (not paired_only) or (paired_only and has_pair):
            # 實例數
            for c, n in class_counts.items():
                stats.per_class_instances[c] += n
                stats.instances_total += n

            # 每張影像的總實例數（僅計 labeled images）
            instances_per_labeled_image.append(valid)

            # 類別涵蓋影像數（每個標註檔的類別集合各加 1）
            for c in cls_set:
                stats.per_class_image_count[c] += 1

    if instances_per_labeled_image:
        stats.instances_per_image_avg = sum(instances_per_labeled_image) / len(instances_per_labeled_image)
        stats.instances_per_image_max = max(instances_per_labeled_image)

    return stats

# --------- 輸出格式 ---------
def format_table(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return ""
    w = max(len(r[0]) for r in rows)
    lines = []
    for k, v in rows:
        lines.append(f"{k.ljust(w)} : {v}")
    return "\n".join(lines)

def print_stats(stats: YOLOStats, class_names: Optional[List[str]] = None):
    print("=== YOLO 資料集統計 ===")
    print(format_table([
        ("影像總數", f"{stats.images_total}"),
        ("標註檔總數", f"{stats.labels_total}"),
        ("有標註影像數", f"{stats.labeled_images}"),
        ("無標註影像數", f"{stats.unlabeled_images}"),
        ("孤兒標註（無對應影像）", f"{stats.labels_without_image}"),
        ("空白標註檔（0 有效行）", f"{stats.empty_label_files}"),
        ("無效標註行總數", f"{stats.invalid_label_lines}"),
        ("實例總數（boxes）", f"{stats.instances_total}"),
        ("每張有標註影像之平均實例數", f"{stats.instances_per_image_avg:.3f}"),
        ("每張有標註影像之最大實例數", f"{stats.instances_per_image_max}"),
    ]))
    print()
    # 影像尺寸分布
    if stats.image_size_counts:
        print("=== 影像尺寸分布（WxH） ===")
        print("image size\tcount")
        for size, cnt in sorted(stats.image_size_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{size}\t{cnt}")
        print()
    else:
        print("（沒有可用的影像尺寸統計；若需啟用請安裝 Pillow：pip install pillow）")
        print()

    # 類別表
    if stats.per_class_instances:
        print("=== 各類別統計 ===")
        print("class_id\tclass_name\tinstances\timages_with_class")
        for cid in sorted(stats.per_class_instances.keys()):
            cname = ""
            if class_names and 0 <= cid < len(class_names):
                cname = class_names[cid]
            print(f"{cid}\t\t{cname}\t\t{stats.per_class_instances[cid]}\t\t{stats.per_class_image_count.get(cid, 0)}")
    else:
        print("（沒有可用的類別實例統計）")

def main():
    parser = argparse.ArgumentParser(
        description="掃描 YOLO 資料集並輸出統計（影像數、標註數、空白標註、各類別實例數等）。"
    )
    parser.add_argument("-r", "--root", required=True, help="資料集根目錄（會被遞迴掃描）")
    parser.add_argument("-e", "--label-ext", default=LABEL_EXT_DEFAULT, help="標註副檔名，預設 .txt")
    parser.add_argument(
        "--img-exts",
        default=",".join(IMG_EXTS_DEFAULT),
        help="影像副檔名清單（逗號分隔），預設：.jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp",
    )
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help="僅統計與影像配對成功的標註（忽略孤兒標註）。",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="顯示掃描進度條（若安裝 tqdm 則使用 tqdm，否則顯示簡易百分比；輸出至 STDERR）。\n建議安裝：pip install tqdm",
    )
    parser.add_argument(
        "--class-names",
        help="以逗號分隔的類別名稱清單（索引對應 class_id）。例如：'person,car,dog'",
    )
    parser.add_argument(
        "--out-json",
        help="將結果另存 JSON 檔路徑（含各類別計數）。",
    )

    args = parser.parse_args()

    dataset_root = os.path.abspath(args.root)
    img_exts = [x.strip() if x.strip().startswith(".") else f".{x.strip()}" for x in args.img_exts.split(",")]

    class_names = None
    if args.class_names:
        class_names = [s.strip() for s in args.class_names.split(",")]

    stats = scan_dataset(
        dataset_root=dataset_root,
        img_exts=img_exts,
        label_ext=args.label_ext,
        paired_only=args.paired_only,
        show_progress=args.progress,
    )

    print_stats(stats, class_names)

    if args.out_json:
        payload = asdict(stats)
        # 將 defaultdict 轉一般 dict
        payload["per_class_instances"] = dict(payload["per_class_instances"])
        payload["per_class_image_count"] = dict(payload["per_class_image_count"])
        payload["image_size_counts"] = dict(payload["image_size_counts"])
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n已輸出 JSON：{args.out_json}")

if __name__ == "__main__":
    main()
