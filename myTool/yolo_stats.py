#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 資料集掃描工具（支援多個 label 副檔名）
- 可統計不同 label-ext 的數量與總數
- 互動式流程的每個步驟都提供取消/返回上層選項
- 維持與原版相容的 CLI（新增 --label-exts；仍支援 -e/--label-ext 可逗號分隔）
"""

import os
import argparse
from collections import defaultdict
from dataclasses import dataclass, asdict, field
import json
import sys
from typing import Dict, List, Optional, Set, Tuple, Iterable
import shutil
import xml.etree.ElementTree as ET

# ---- 第三方（可選） ----
try:
    from PIL import Image  # type: ignore
except Exception:
    Image = None  # pillow 未安裝時允許降級

try:
    from tqdm.auto import tqdm as _tqdm  # type: ignore
except Exception:
    _tqdm = None

# ---- 常數 ----
IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
LABEL_EXTS_DEFAULT = [".txt"]


# =========================
# 公用：進度列
# =========================
def _progress_iter(iterable: Iterable, total: Optional[int], desc: str, enabled: bool):
    """包裝 iterable：enabled=False 直接回傳；若可用 tqdm 用 tqdm；否則 STDERR 簡易百分比"""
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


# =========================
# 路徑與配對輔助
# =========================
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


def find_label_for_image(image_path: str, label_exts: List[str]) -> Optional[str]:
    """由影像推測標註檔路徑：同目錄；或 images -> labels/label。支援多個 label_exts。"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    img_dir = os.path.dirname(image_path)

    # 同目錄
    for ext in label_exts:
        cand = os.path.join(img_dir, stem + ext)
        if os.path.isfile(cand):
            return cand

    # images -> labels
    cand_dir = _replace_one_component(img_dir, "images", "labels")
    if cand_dir:
        for ext in label_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    # images -> label
    cand_dir = _replace_one_component(img_dir, "images", "label")
    if cand_dir:
        for ext in label_exts:
            cand = os.path.join(cand_dir, stem + ext)
            if os.path.isfile(cand):
                return cand

    return None


# =========================
# 標註解析
# =========================
def parse_yolo_label_file(path: str) -> Tuple[int, Dict[int, int], Set[int], int]:
    """
    讀取 YOLO 標註檔：
    回傳 (有效行數, 各類別實例數, 本檔包含之類別集合, 無效行數)
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


def parse_voc_xml_label_file(path: str) -> Tuple[int, Dict[int, int], Set[int], int]:
    """解析 VOC XML；以 <object> 數量作為有效實例數。若解析失敗，計 1 無效行。"""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        # VOC 結構通常為 <annotation><object>...</object></annotation>
        objects = root.findall('.//object')
        valid = len(objects)
        # 這裡不蒐集類別名稱到 per_class（維持 YOLO 整數 ID 統計的既有行為）
        return valid, defaultdict(int), set(), 0
    except Exception:
        # 解析失敗視為 1 無效行，避免中斷
        return 0, defaultdict(int), set(), 1


def parse_label_file(path: str) -> Tuple[int, Dict[int, int], Set[int], int]:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.xml':
        return parse_voc_xml_label_file(path)
    # 預設以 YOLO 純文字格式解析
    return parse_yolo_label_file(path)


# =========================
# 影像尺寸
# =========================
def get_image_size(path: str) -> Optional[Tuple[int, int]]:
    """回傳 (width, height)。若未安裝 Pillow 或讀取失敗則回傳 None。"""
    if Image is None:
        return None
    try:
        with Image.open(path) as im:
            return im.size  # (w, h)
    except Exception:
        return None


# =========================
# 統計資料結構
# =========================
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

    # 新增：label 副檔名分布
    labels_total_by_ext: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    empty_label_files_by_ext: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # 清單（互動用）
    image_paths: List[str] = field(default_factory=list)
    label_paths: List[str] = field(default_factory=list)
    labeled_image_paths: List[str] = field(default_factory=list)
    unlabeled_image_paths: List[str] = field(default_factory=list)
    labels_without_image_paths: List[str] = field(default_factory=list)
    empty_label_file_paths: List[str] = field(default_factory=list)


# =========================
# 設定（統一狀態來源）
# =========================
@dataclass
class Config:
    root: str
    label_exts: List[str] = field(default_factory=lambda: LABEL_EXTS_DEFAULT.copy())
    img_exts: List[str] = field(default_factory=lambda: IMG_EXTS_DEFAULT.copy())
    paired_only: bool = False
    progress: bool = True
    class_names: Optional[List[str]] = None

    @staticmethod
    def normalize_exts(exts_in: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in exts_in:
            s = str(raw).strip()
            if not s:
                continue
            if not s.startswith("."):
                s = "." + s
            s = s.lower()
            if s not in seen:
                seen.add(s)
                out.append(s)
        return out

    @property
    def img_exts_str(self) -> str:
        return ",".join(self.img_exts)

    @property
    def label_exts_str(self) -> str:
        return ",".join(self.label_exts)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        # label exts 來源優先順序：--label-exts > -e/--label-ext
        label_exts_raw = None
        if getattr(args, "label_exts", None):
            label_exts_raw = args.label_exts
        elif getattr(args, "label_ext", None):
            label_exts_raw = args.label_ext
        else:
            label_exts_raw = ",".join(LABEL_EXTS_DEFAULT)

        label_list = [x.strip() for x in str(label_exts_raw).split(",")]
        img_list = [x.strip() for x in str(args.img_exts).split(",")]

        return cls(
            root=os.path.abspath(args.root),
            label_exts=cls.normalize_exts(label_list),
            img_exts=cls.normalize_exts(img_list),
            paired_only=bool(args.paired_only),
            progress=(not getattr(args, "no_progress", False)),
            class_names=[s.strip() for s in args.class_names.split(",")] if args.class_names else None,
        )


# =========================
# 掃描主流程（純函式）
# =========================
def scan_dataset(
    dataset_root: str,
    img_exts: List[str],
    label_exts: List[str],
    paired_only: bool = False,
    show_progress: bool = False,
) -> YOLOStats:
    stats = YOLOStats()

    # 收集所有影像與標註檔（支援多個 label 副檔名）
    images: List[str] = []
    labels: List[str] = []
    for root, _, files in os.walk(dataset_root):
        for fname in files:
            fpath = os.path.join(root, fname)
            low = fname.lower()
            if any(low.endswith(ext) for ext in img_exts):
                images.append(fpath)
            else:
                for lext in label_exts:
                    if low.endswith(lext):
                        labels.append(fpath)
                        stats.labels_total_by_ext[lext] += 1
                        break

    stats.images_total = len(images)
    stats.labels_total = len(labels)
    stats.image_paths = images
    stats.label_paths = labels

    # 影像：統計尺寸＆是否有標註
    for img in _progress_iter(images, total=len(images), desc="掃描影像（尺寸與標註配對）", enabled=show_progress):
        if Image is not None:
            sz = get_image_size(img)
            if sz:
                w, h = sz
                stats.image_size_counts[f"{w}x{h}"] += 1
        lab = find_label_for_image(img, label_exts)
        if lab:
            stats.labeled_image_paths.append(img)
        else:
            stats.unlabeled_image_paths.append(img)

    stats.labeled_images = len(stats.labeled_image_paths)
    stats.unlabeled_images = len(stats.unlabeled_image_paths)

    # 標註：解析有效行、孤兒、空白（統計空白檔的 ext 分布）
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

        valid, class_counts, cls_set, invalid_lines = parse_label_file(lab)
        stats.invalid_label_lines += invalid_lines
        if valid == 0:
            stats.empty_label_files += 1
            stats.empty_label_file_paths.append(lab)
            # 依副檔名記錄空白檔
            ext = os.path.splitext(lab)[1].lower()
            stats.empty_label_files_by_ext[ext] += 1

        if (not paired_only) or (paired_only and has_pair):
            for c, n in class_counts.items():
                stats.per_class_instances[c] += n
                stats.instances_total += n
            instances_per_labeled_image.append(valid)
            for c in cls_set:
                stats.per_class_image_count[c] += 1

    if instances_per_labeled_image:
        stats.instances_per_image_avg = sum(instances_per_labeled_image) / len(instances_per_labeled_image)
        stats.instances_per_image_max = max(instances_per_labeled_image)

    return stats


# =========================
# 輸出
# =========================
def format_table(rows: List[Tuple[str, str]]) -> str:
    if not rows:
        return ""
    w = max(len(r[0]) for r in rows)
    return "\n".join(f"{k.ljust(w)} : {v}" for k, v in rows)


def print_stats(stats: YOLOStats, class_names: Optional[List[str]] = None):
    print("=== YOLO 資料集統計 ===")
    print(
        format_table(
            [
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
            ]
        )
    )
    print()

    # 影像尺寸分布
    if stats.image_size_counts:
        print("=== 影像尺寸分布（WxH） ===")
        print("image size\tcount")
        for size, cnt in sorted(stats.image_size_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"{size}\t{cnt}")
        print()
    else:
        print("（沒有可用的影像尺寸統計；若需啟用請安裝 Pillow：pip install pillow）\n")

    # 標註副檔名分布
    if stats.labels_total_by_ext:
        print("=== 標註副檔名分布 ===")
        print("label_ext\tcount\tempty_files")
        for ext, cnt in sorted(stats.labels_total_by_ext.items(), key=lambda kv: (-kv[1], kv[0])):
            empty_cnt = stats.empty_label_files_by_ext.get(ext, 0)
            print(f"{ext}\t{cnt}\t{empty_cnt}")
        print()

    # 類別表
    if stats.per_class_instances:
        print("=== 各類別統計 ===")
        print("class_id\tclass_name\tinstances\timages_with_class")
        for cid in sorted(stats.per_class_instances.keys()):
            cname = ""
            if class_names and 0 <= cid < len(class_names):
                cname = class_names[cid]
            print(
                f"{cid}\t\t{cname}\t\t{stats.per_class_instances[cid]}\t\t{stats.per_class_image_count.get(cid, 0)}"
            )
    else:
        print("（沒有可用的類別實例統計）")


# =========================
# 批次處理
# =========================
def _ensure_nested_target(dst_root: str, dataset_root: str, src_path: str) -> str:
    """在 dst_root 內建立相對於 dataset_root 的巢狀結構並回傳目標路徑"""
    rel = os.path.relpath(src_path, start=dataset_root)
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
            if not os.path.exists(p):
                raise FileNotFoundError("source not found")

            if action in ("copy", "move"):
                if not dst_root:
                    raise ValueError("dst_root 未指定")
                out_path = _ensure_nested_target(dst_root, dataset_root, p)
                out_dir = os.path.dirname(out_path)
                if not dry_run:
                    os.makedirs(out_dir, exist_ok=True)
                    if action == "copy":
                        shutil.copy2(p, out_path)
                    else:
                        shutil.move(p, out_path)
            elif action == "delete":
                if not dry_run:
                    os.remove(p)
            else:
                raise ValueError(f"未知的動作: {action}")

            ok += 1
        except Exception as e:
            print(f"[!] 無法處理：{p} ({e})", file=sys.stderr)
            fail += 1
    return ok, fail


# =========================
# 互動式應用（統一流程，處處可取消/返回）
# =========================
CANCEL_TOKENS = {"q", "Q", "b", "B", "back", "cancel"}


def _is_cancel(s: str) -> bool:
    return s.strip() in CANCEL_TOKENS


def _input_line(prompt: str, allow_empty: bool = False, show_cancel_hint: bool = True) -> Optional[str]:
    """輸入一行文字；若輸入 q/b/cancel 則回傳 None 表取消/返回。"""
    hint = "  (輸入 q 取消/返回)" if show_cancel_hint else ""
    while True:
        s = input(prompt + hint + " ").strip()
        if _is_cancel(s):
            return None
        if s == "" and not allow_empty:
            print("輸入不可為空，或輸入 q 返回。")
            continue
        return s


def _input_nonempty(prompt: str) -> Optional[str]:
    return _input_line(prompt, allow_empty=False)


def _ask_yes_no(prompt: str, default: bool = True) -> Optional[bool]:
    suffix = " [Y/n] " if default else " [y/N] "
    s = input(prompt + suffix + "(輸入 q 取消/返回) ").strip().lower()
    if _is_cancel(s):
        return None
    if s == "":
        return default
    return s in ("y", "yes")


def _choose_category(stats: YOLOStats) -> Optional[Tuple[str, List[str]]]:
    print("\n可選清單類別：")
    print(f"  1) 無標註影像（{len(stats.unlabeled_image_paths)}）")
    print(f"  2) 孤兒標註（{len(stats.labels_without_image_paths)}）")
    print(f"  3) 空白標註檔（{len(stats.empty_label_file_paths)}）")
    print("  0) 返回上層")
    mapping = {
        "1": ("unlabeled_images", stats.unlabeled_image_paths),
        "2": ("orphan_labels", stats.labels_without_image_paths),
        "3": ("empty_labels", stats.empty_label_file_paths),
    }
    while True:
        c = input("請選擇類別 (1/2/3 或 0 返回)：").strip()
        if _is_cancel(c) or c == "0":
            return None
        if c in mapping:
            return mapping[c]
        print("無效選擇，請重試。")


class InteractiveApp:
    """把互動邏輯集中於此，確保所有重新掃描走同一路徑，且每一步可取消/返回。"""

    def __init__(self, config: Config, initial_stats: YOLOStats):
        self.config = config
        self.stats = initial_stats  # 最近一次掃描結果

    def rescan(self):
        self.stats = scan_dataset(
            dataset_root=self.config.root,
            img_exts=self.config.img_exts,
            label_exts=self.config.label_exts,
            paired_only=self.config.paired_only,
            show_progress=self.config.progress,
        )

    def show_menu(self):
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
                print_stats(self.stats, self.config.class_names)

            elif choice == "2":
                chosen = _choose_category(self.stats)
                if not chosen:
                    continue
                cat_name, paths = chosen
                print(f"\n=== 類別：{cat_name}，共 {len(paths)} 個 ===")
                preview = min(20, len(paths))
                for i, p in enumerate(paths[:preview], 1):
                    print(f"{i:>3}: {p}")
                if len(paths) > preview:
                    print(f"...（其餘 {len(paths) - preview} 筆未列出）")

            elif choice == "3":
                chosen = _choose_category(self.stats)
                if not chosen:
                    continue
                cat_name, paths = chosen
                if not paths:
                    print("該類別目前為空，無需處理。")
                    continue
                print(f"選擇的清單：{cat_name}，共 {len(paths)} 筆")

                print("選擇動作：c=copy / m=move / d=delete / b=返回上層")
                action = None
                while True:
                    a = input("要執行的動作？：").strip().lower()
                    if _is_cancel(a) or a == "b":
                        action = None
                        break
                    if a in {"c", "m", "d"}:
                        action = {"c": "copy", "m": "move", "d": "delete"}[a]
                        break
                    print("無效動作，請重試。")
                if action is None:
                    continue

                dst_root = None
                if action in ("copy", "move"):
                    dst_root = _input_nonempty("請輸入目標根目錄（會建立巢狀副本）:")
                    if dst_root is None:
                        # 取消
                        continue
                    print(f"將在 {dst_root} 下建立相對於資料集 root 的巢狀結構。")

                limit_str = _input_line("只處理前 N 筆？(Enter=全部)", allow_empty=True)
                if limit_str is None:
                    continue
                if limit_str.isdigit():
                    paths_to_do = paths[: int(limit_str)]
                else:
                    paths_to_do = paths

                yn = _ask_yes_no("先 Dry-Run 模擬？", default=True)
                if yn is None:
                    continue
                dry_run = yn

                ok, fail = batch_process_paths(paths_to_do, action, self.config.root, dst_root, dry_run=dry_run)
                print(("Dry-Run 結果：" if dry_run else "實際執行結果：") + f"成功 {ok}，失敗 {fail}")

                if dry_run:
                    yn2 = _ask_yes_no("要依同設定真的執行嗎？", default=False)
                    if yn2 is None:
                        continue
                    if yn2:
                        ok, fail = batch_process_paths(paths_to_do, action, self.config.root, dst_root, dry_run=False)
                        print(f"實際執行完成：成功 {ok}，失敗 {fail}")

                print("提醒：若有移動/刪除，建議執行『重新掃描』更新統計。")

            elif choice == "4":
                print("重新掃描（沿用目前參數）...")
                self.rescan()
                print_stats(self.stats, self.config.class_names)

            elif choice == "5":
                # 修改設定（統一正規化）—任一步驟可取消/返回
                new_root_in = _input_line("新 root 路徑（Enter 保持不變 / q 返回）：", allow_empty=True)
                if new_root_in is None:
                    continue
                if new_root_in == "":
                    new_root = self.config.root
                else:
                    new_root = os.path.abspath(new_root_in)

                new_label_exts_in = _input_line(
                    f"label 副檔名清單(逗號分隔；目前 {self.config.label_exts_str}，Enter 保持)：",
                    allow_empty=True,
                )
                if new_label_exts_in is None:
                    continue
                new_label_exts = (
                    Config.normalize_exts(new_label_exts_in.split(",")) if new_label_exts_in else self.config.label_exts
                )

                new_img_exts_in = _input_line(
                    f"影像副檔名清單(逗號分隔；目前 {self.config.img_exts_str}，Enter 保持)：",
                    allow_empty=True,
                )
                if new_img_exts_in is None:
                    continue
                new_img_exts = (
                    Config.normalize_exts(new_img_exts_in.split(",")) if new_img_exts_in else self.config.img_exts
                )

                yn_paired = _ask_yes_no("只統計有配對的標註？", default=self.config.paired_only)
                if yn_paired is None:
                    continue
                # 套用
                self.config.root = new_root
                self.config.label_exts = new_label_exts
                self.config.img_exts = new_img_exts
                self.config.paired_only = yn_paired
                # 進度列預設啟用，互動流程不再詢問；如需停用請於 CLI 使用 --no-progress

                print("重新掃描（修改後參數）...")
                self.rescan()
                print_stats(self.stats, self.config.class_names)

            elif choice == "0":
                print("已離開。")
                break
            else:
                print("無效選項，請重試。")


# =========================
# CLI 入口
# =========================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "掃描 YOLO 資料集並輸出統計（影像數、標註數、空白標註、各類別實例數等）。支援互動式批次處理，"
            "並支援多個 label 副檔名。"
        )
    )
    p.add_argument("-r", "--root", required=True, help="資料集根目錄（會被遞迴掃描）")
    # 新增 --label-exts；仍保留 -e/--label-ext 相容（都可逗號分隔）。如同時提供，以 --label-exts 優先。
    p.add_argument(
        "--label-exts",
        help="標註副檔名清單（逗號分隔），例如 .txt,.seg；優先於 -e/--label-ext",
    )
    p.add_argument(
        "-e",
        "--label-ext",
        default=",".join(LABEL_EXTS_DEFAULT),
        help="標註副檔名（或以逗號分隔多個），預設 .txt",
    )
    p.add_argument(
        "--img-exts",
        default=",".join(IMG_EXTS_DEFAULT),
        help="影像副檔名清單（逗號分隔），預設：.jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp",
    )
    p.add_argument(
        "--paired-only",
        action="store_true",
        help="僅統計與影像配對成功的標註（忽略孤兒標註）。",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="（已預設啟用）顯示掃描進度條；未安裝 tqdm 時會退回為簡易百分比。",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="停用進度列（預設啟用；未安裝 tqdm 時退回為簡易百分比）。",
    )
    p.add_argument(
        "--class-names",
        help="以逗號分隔的類別名稱清單（索引對應 class_id）。例如：'person,car,dog'",
    )
    p.add_argument(
        "--out-json",
        help="將結果另存 JSON 檔路徑（含各類別與副檔名計數）。",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="非互動模式：只掃描一次並輸出統計後結束（相容舊流程）。",
    )
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1) 解析成 Config（一次正規化，之後都用同一份狀態）
    config = Config.from_args(args)

    # 2) 首次掃描 + 輸出
    stats = scan_dataset(
        dataset_root=config.root,
        img_exts=config.img_exts,
        label_exts=config.label_exts,
        paired_only=config.paired_only,
        show_progress=config.progress,
    )
    print_stats(stats, config.class_names)

    # 3) 可選 JSON 輸出（處理 defaultdict）
    if args.out_json:
        payload = asdict(stats)
        payload["per_class_instances"] = dict(payload["per_class_instances"])  # type: ignore
        payload["per_class_image_count"] = dict(payload["per_class_image_count"])  # type: ignore
        payload["image_size_counts"] = dict(payload["image_size_counts"])  # type: ignore
        payload["labels_total_by_ext"] = dict(payload["labels_total_by_ext"])  # type: ignore
        payload["empty_label_files_by_ext"] = dict(payload["empty_label_files_by_ext"])  # type: ignore
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n已輸出 JSON：{args.out_json}")

    # 4) 互動模式
    if not args.no_interactive:
        app = InteractiveApp(config=config, initial_stats=stats)
        app.show_menu()


if __name__ == "__main__":
    main()
