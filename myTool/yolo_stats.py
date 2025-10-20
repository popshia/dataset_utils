#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict, field
import json
import sys
from typing import Dict, List, Optional, Set, Tuple
import shutil

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
        yield from _tqdm(iterable, total=total, desc=desc, unit="file")
    else:
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
    per_class_instances: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    per_class_image_count: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    image_size_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # ---- 新增：用於互動後處理的清單 ----
    image_paths: List[str] = field(default_factory=list)
    label_paths: List[str] = field(default_factory=list)
    labeled_image_paths: List[str] = field(default_factory=list)
    unlabeled_image_paths: List[str] = field(default_factory=list)
    labels_without_image_paths: List[str] = field(default_factory=list)
    empty_label_file_paths: List[str] = field(default_factory=list)


# --------- 主流程：掃描 ---------
def scan_dataset(
    dataset_root: str,
    img_exts: List[str],
    label_ext: str,
    paired_only: bool = False,
    show_progress: bool = False,
) -> YOLOStats:
    stats = YOLOStats()

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
    stats.image_paths = images
    stats.label_paths = labels

    # 影像：統計尺寸＆是否有標註
    for img in _progress_iter(images, total=len(images), desc="掃描影像（尺寸與標註配對）", enabled=show_progress):
        # 尺寸
        if Image is not None:
            sz = get_image_size(img)
            if sz:
                w, h = sz
                stats.image_size_counts[f"{w}x{h}"] += 1
        # 配對
        lab = find_label_for_image(img, label_ext)
        if lab:
            stats.labeled_image_paths.append(img)
        else:
            stats.unlabeled_image_paths.append(img)

    stats.labeled_images = len(stats.labeled_image_paths)
    stats.unlabeled_images = len(stats.unlabeled_image_paths)

    # 標註：解析有效行、孤兒、空白
    instances_per_labeled_image: List[int] = []
    for lab in _progress_iter(labels, total=len(labels), desc="解析標註檔", enabled=show_progress):
        img = find_image_for_label(lab, img_exts)
        has_pair = img is not None
        if not has_pair:
            stats.labels_without_image += 1
            stats.labels_without_image_paths.append(lab)
            if paired_only:
                # 僅統計有配對者時，遇到孤兒標註就跳過解析
                continue

        valid, class_counts, cls_set, invalid_lines = parse_yolo_label_file(lab)
        stats.invalid_label_lines += invalid_lines
        if valid == 0:
            stats.empty_label_files += 1
            stats.empty_label_file_paths.append(lab)

        if (not paired_only) or (paired_only and has_pair):
            # 累計實例數
            for c, n in class_counts.items():
                stats.per_class_instances[c] += n
                stats.instances_total += n
            # 每張有標註影像的實例數
            instances_per_labeled_image.append(valid)
            # 出現該類別的影像數（依標註檔的類別集合）
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


# --------- 批次處理（複製 / 移動 / 刪除） ---------
def _ensure_nested_target(dst_root: str, dataset_root: str, src_path: str) -> str:
    """回傳欲輸出之目標完整路徑（於 dst_root 內建立相對於 dataset_root 的巢狀結構）。"""
    rel = os.path.relpath(src_path, start=dataset_root)
    # 防止 rel 以 ".." 跳出
    if rel.startswith(".."):
        rel = os.path.basename(src_path)
    return os.path.join(dst_root, rel)

def batch_process_paths(
    paths: List[str],
    action: str,
    dataset_root: str,
    dst_root: Optional[str] = None,
    dry_run: bool = True,
) -> Tuple[int, int]:
    """
    對 paths 進行批次處理：
    - action: 'copy' | 'move' | 'delete'
    - 複製/移動：在 dst_root 生成「巢狀副本」
    - 回傳 (成功數, 失敗數)
    """
    ok = 0
    fail = 0
    for p in paths:
        try:
            if action in ("copy", "move"):
                if not dst_root:
                    raise ValueError("dst_root 未指定")
                out_path = _ensure_nested_target(dst_root, dataset_root, p)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if dry_run:
                    # 不實作
                    pass
                else:
                    if action == "copy":
                        shutil.copy2(p, out_path)
                    else:
                        shutil.move(p, out_path)
            elif action == "delete":
                if dry_run:
                    pass
                else:
                    os.remove(p)
            else:
                raise ValueError(f"未知的動作: {action}")
            ok += 1
        except Exception as e:
            print(f"[!] 無法處理：{p} ({e})", file=sys.stderr)
            fail += 1
    return ok, fail


# --------- 互動式選單 ---------
def _input_nonempty(prompt: str) -> str:
    while True:
        s = input(prompt).strip()
        if s:
            return s

def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    s = input(prompt + suffix).strip().lower()
    if s == "":
        return default
    return s in ("y", "yes")

def _choose_category(stats: YOLOStats) -> Tuple[str, List[str]]:
    print("\n可選清單類別：")
    print(f"  1) 無標註影像（{len(stats.unlabeled_image_paths)}）")
    print(f"  2) 孤兒標註（{len(stats.labels_without_image_paths)}）")
    print(f"  3) 空白標註檔（{len(stats.empty_label_file_paths)}）")
    mapping = {
        "1": ("unlabeled_images", stats.unlabeled_image_paths),
        "2": ("orphan_labels", stats.labels_without_image_paths),
        "3": ("empty_labels", stats.empty_label_file_paths),
    }
    while True:
        c = input("請選擇類別 (1/2/3)：").strip()
        if c in mapping:
            return mapping[c]
        print("無效選擇，請重試。")

def interactive_loop(dataset_root: str, args, stats: YOLOStats, class_names: Optional[List[str]]):
    last_stats = stats
    while True:
        print("\n========== 互動選單 ==========")
        print("1) 查看統計")
        print("2) 列出某類清單（無標註影像 / 孤兒標註 / 空白標註）")
        print("3) 對某類清單進行批次處理（複製 / 移動 / 刪除）")
        print("4) 重新掃描（沿用目前參數）")
        print("5) 重新掃描（修改參數）")
        print("0) 離開")
        choice = input("請輸入選項：").strip()

        if choice == "1":
            print_stats(last_stats, class_names)

        elif choice == "2":
            cat_name, paths = _choose_category(last_stats)
            print(f"\n=== 類別：{cat_name}，共 {len(paths)} 個 ===")
            preview = min(20, len(paths))
            for i, p in enumerate(paths[:preview], 1):
                print(f"{i:>3}: {p}")
            if len(paths) > preview:
                print(f"...（其餘 {len(paths) - preview} 筆未列出）")

        elif choice == "3":
            cat_name, paths = _choose_category(last_stats)
            if not paths:
                print("該類別目前為空，無需處理。")
                continue
            print(f"選擇的清單：{cat_name}，共 {len(paths)} 筆")
            action_map = {"c": "copy", "m": "move", "d": "delete"}
            while True:
                a = input("要執行的動作？ (c=copy / m=move / d=delete)：").strip().lower()
                if a in action_map:
                    action = action_map[a]
                    break
                print("無效動作，請重試。")

            dst_root = None
            if action in ("copy", "move"):
                dst_root = _input_nonempty("請輸入目標根目錄（會建立巢狀副本）：")
                print(f"將在 {dst_root} 下建立相對於資料集 root 的巢狀結構。")

            # 可選：限制處理筆數
            limit_str = input("只處理前 N 筆？(Enter=全部)：").strip()
            if limit_str.isdigit():
                paths_to_do = paths[: int(limit_str)]
            else:
                paths_to_do = paths

            dry_run = _ask_yes_no("先 Dry-Run 模擬？", default=True)
            ok, fail = batch_process_paths(paths_to_do, action, dataset_root, dst_root, dry_run=dry_run)
            print(f"Dry-Run 結果：" if dry_run else "實際執行結果：", end="")
            print(f"成功 {ok}，失敗 {fail}")

            if dry_run and _ask_yes_no("要依同設定真的執行嗎？", default=False):
                ok, fail = batch_process_paths(paths_to_do, action, dataset_root, dst_root, dry_run=False)
                print(f"實際執行完成：成功 {ok}，失敗 {fail}")

            print("提醒：若有移動/刪除，建議執行『重新掃描』更新統計。")

        elif choice == "4":
            print("重新掃描（沿用目前參數）...")
            last_stats = scan_dataset(
                dataset_root=dataset_root,
                img_exts=[x.strip() if x.strip().startswith(".") else f".{x.strip()}" for x in args.img_exts.split(",")],
                label_ext=args.label_ext,
                paired_only=args.paired_only,
                show_progress=args.progress,
            )
            print_stats(last_stats, class_names)

        elif choice == "5":
            # 修改參數後重掃
            dataset_root = os.path.abspath(_input_nonempty("新 root 路徑："))
            label_ext = input(f"標註副檔名(預設 {args.label_ext})：").strip() or args.label_ext
            img_exts_in = input(f"影像副檔名清單(逗號分隔；預設 {args.img_exts})：").strip() or args.img_exts
            paired_only = _ask_yes_no("只統計有配對的標註？", default=args.paired_only)
            progress = _ask_yes_no("顯示進度列？", default=args.progress)

            args.root = dataset_root
            args.label_ext = label_ext
            args.img_exts = img_exts_in
            args.paired_only = paired_only
            args.progress = progress

            print("重新掃描（修改後參數）...")
            last_stats = scan_dataset(
                dataset_root=dataset_root,
                img_exts=[x.strip() if x.strip().startswith(".") else f".{x.strip()}" for x in img_exts_in.split(",")],
                label_ext=label_ext,
                paired_only=paired_only,
                show_progress=progress,
            )
            print_stats(last_stats, class_names)

        elif choice == "0":
            print("已離開。")
            break
        else:
            print("無效選項，請重試。")


# --------- CLI 入口 ---------
def main():
    parser = argparse.ArgumentParser(
        description="掃描 YOLO 資料集並輸出統計（影像數、標註數、空白標註、各類別實例數等）。支援互動式批次處理。"
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
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="非互動模式：只掃描一次並輸出統計後結束（相容舊流程）。",
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

    if not args.no_interactive:
        # 進入互動式迴圈
        interactive_loop(dataset_root, args, stats, class_names)


if __name__ == "__main__":
    main()
