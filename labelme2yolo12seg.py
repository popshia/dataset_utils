#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labelme2yolo12seg.py
--------------------
將 Labelme 的 JSON 標註轉為「Ultralytics YOLO 分割格式」（YOLOv8/YOLOv11/YOLOv12 共用），
並自動建立資料夾結構與 dataset.yaml。

使用方式
========
1) 準備一個 names.txt（每行一個類別名稱；允許空行與以 # 開頭的註解行）
   例如：
       person
       dog
       cat

2) 執行：
   python labelme2yolo12seg.py \
       --input /path/to/labelme_dir \
       --output /path/to/output_dataset \
       --names /path/to/names.txt \
       --val-ratio 0.2 --test-ratio 0.0 --copy

   * 預設會嘗試建立 symlink 以節省空間（Windows 可能需要管理員權限）；加上 --copy 會改為複製檔案。
   * 會把資料隨機切成 train/val(/test) 並在 labels/ 下輸出 .txt；images/ 下放影像。

支援要點
========
- 支援 Labelme 的 polygon 與 rectangle（會自動轉成四邊形多邊形）；其他 shape_type 會略過（circle/line/point 等）。
- 若 JSON 指到的影像不存在，但 JSON 內含 imageData，將自動解碼存檔。
- 若某張影像沒有任何可用標註，會產生「空的 .txt」以方便訓練管線處理（可用 --skip-empty-label 禁用）。
- 將座標正規化至 0~1，並做輕微邊界裁切避免超界。

輸出結構
========
output_dataset/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/            # 若 test-ratio > 0 才會建立
├─ labels/
│  ├─ train/
│  ├─ val/
│  └─ test/            # 若 test-ratio > 0 才會建立
└─ dataset.yaml           # 可直接給 Ultralytics `yolo segment train` 使用

"""

import argparse
import base64
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

EPS = 1e-6


def read_names(names_path: Path) -> List[str]:
    """讀取 names.txt，一行一個類別；忽略空白行與 # 註解行。"""
    names: List[str] = []
    with names_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    if not names:
        raise ValueError(f"names.txt 內容為空：{names_path}")
    return names


def build_label_to_id(names: List[str]) -> Dict[str, int]:
    """建立 label -> class_id 對照表（依 names 順序，從 0 起算）。"""
    return {name: i for i, name in enumerate(names)}


def find_image_for_json(json_data: dict, json_path: Path, img_exts: List[str]) -> Path:
    """
    根據 Labelme JSON 的 imagePath 以及 json 檔名尋找影像檔。
    若 imagePath 是相對路徑，會相對於 json_path.parent。
    若找不到，會再嘗試以 json 同名不同副檔名搜尋。
    """
    # 1) 尊重 imagePath
    img_path_field = json_data.get("imagePath")
    if img_path_field:
        p = (json_path.parent / img_path_field).resolve()
        if p.exists():
            return p

    # 2) 同名搜尋
    stem = json_path.stem
    for ext in img_exts:
        p = (json_path.parent / f"{stem}{ext}").resolve()
        if p.exists():
            return p

    # 3) 都找不到，回傳預期位置（呼叫端可決定是否要用 imageData 生成）
    return (json_path.parent / (img_path_field or f"{stem}{img_exts[0]}")).resolve()


def maybe_write_image_from_imagedata(json_data: dict, img_path: Path) -> bool:
    """
    若 image 檔案不存在但 JSON 內有 imageData，則解碼存成 img_path。
    回傳是否有成功寫入。
    """
    if img_path.exists():
        return True
    image_data = json_data.get("imageData")
    if not image_data:
        return False
    try:
        # imageData 可能是 base64（字串），也可能是 dataURI。
        if "," in image_data and image_data.strip().startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        raw = base64.b64decode(image_data)
        img_path.parent.mkdir(parents=True, exist_ok=True)
        with img_path.open("wb") as f:
            f.write(raw)
        return True
    except Exception as e:
        print(f"[WARN] 解析 imageData 失敗：{img_path} ({e})")
        return False


def shape_to_polygon(shape: dict) -> Optional[List[Tuple[float, float]]]:
    """
    從 Labelme shape 取出 polygon 頂點；
    - polygon: 直接回傳 points
    - rectangle: 兩點 -> 轉成四點矩形
    其他類型（circle/line/point 等）回傳 None 以略過。
    """
    st = shape.get("shape_type", "polygon")
    pts = shape.get("points", [])
    if st == "polygon":
        if len(pts) >= 3:
            return [(float(x), float(y)) for x, y in pts]
        return None
    if st == "rectangle":
        if len(pts) != 2:
            return None
        (x1, y1), (x2, y2) = pts
        x_min, x_max = sorted([float(x1), float(x2)])
        y_min, y_max = sorted([float(y1), float(y2)])
        return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
    return None  # 不支援的 shape_type


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_polygon(poly: List[Tuple[float, float]], w: int, h: int) -> List[Tuple[float, float]]:
    """將像素座標正規化到 0~1，並稍作裁切避免等於 1 或小於 0。"""
    out = []
    for x, y in poly:
        x = _clip(x, 0.0, w - EPS)
        y = _clip(y, 0.0, h - EPS)
        out.append((x / max(1.0, float(w)), y / max(1.0, float(h))))
    return out


def remove_near_duplicate_points(poly: List[Tuple[float, float]], tol: float = 0.0) -> List[Tuple[float, float]]:
    """移除連續重複點；若 tol > 0 則移除距離極小的點（避免退化多邊形）。"""
    if not poly:
        return poly
    cleaned = [poly[0]]
    for x, y in poly[1:]:
        px, py = cleaned[-1]
        if tol > 0:
            if (x - px) ** 2 + (y - py) ** 2 <= tol * tol:
                continue
        else:
            if x == px and y == py:
                continue
        cleaned.append((x, y))
    # 若首尾重複也移除
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def polygon_to_yolo_line(class_id: int, poly01: List[Tuple[float, float]]) -> Optional[str]:
    """將 0~1 座標的多邊形轉成 YOLO segmentation 一行字串。"""
    if len(poly01) < 3:
        return None
    coords = []
    for x, y in poly01:
        # 防止 -0.0
        x = 0.0 if abs(x) < EPS else x
        y = 0.0 if abs(y) < EPS else y
        coords.append(f"{x:.6f} {y:.6f}")
    return f"{class_id} " + " ".join(coords)


def symlink_or_copy(src: Path, dst: Path, force_copy: bool = False) -> None:
    """在可能的情況下建立 symlink；否則回退到複製。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if not force_copy:
        try:
            # 使用相對 symlink 以方便搬移 output 資料夾
            rel = os.path.relpath(src, start=dst.parent)
            os.symlink(rel, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def write_data_yaml(out_dir: Path, names: List[str], has_test: bool) -> Path:
    """建立 Ultralytics dataset.yaml。"""
    yaml_path = out_dir / "dataset.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        train_p = (out_dir / "images/train").resolve().as_posix()
        val_p   = (out_dir / "images/val").resolve().as_posix()
        f.write(f'train: "{train_p}"\n')
        f.write(f'val: "{val_p}"\n')
        if has_test:
            test_p = (out_dir / "images/test").resolve().as_posix()
            f.write(f'test: "{test_p}"\n')
        f.write("names:\n")
        for i, name in enumerate(names):
            safe = str(name).replace(":", "_")
            f.write(f"  {i}: {safe}\n")
    return yaml_path


def collect_items(input_dir: Path, names: List[str], img_exts: List[str],
                  skip_empty_label: bool) -> Tuple[List[Tuple[Path, List[str]]], List[str]]:
    """
    讀取 input_dir 下所有 .json，回傳 (items, warnings)
    items: List of (image_path, label_lines)
    """
    label_to_id = build_label_to_id(names)
    items: List[Tuple[Path, List[str]]] = []
    warnings: List[str] = []

    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"找不到任何 JSON：{input_dir}")

    for jpath in json_files:
        try:
            with jpath.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            warnings.append(f"[WARN] 無法解析 JSON：{jpath} ({e})")
            continue

        iw, ih = data.get("imageWidth"), data.get("imageHeight")
        if not isinstance(iw, int) or not isinstance(ih, int):
            # 有些標註沒有寫入寬高；如果影像存在就讀取實際大小。
            img_probe = find_image_for_json(data, jpath, img_exts)
            if img_probe.exists():
                try:
                    from PIL import Image  # 可選；若未安裝則略過
                    with Image.open(img_probe) as im:
                        iw, ih = im.size
                except Exception:
                    pass
        if not isinstance(iw, int) or not isinstance(ih, int):
            warnings.append(f"[WARN] 缺少 imageWidth/Height，且無法偵測：{jpath}")
            continue

        img_path = find_image_for_json(data, jpath, img_exts)
        if not img_path.exists():
            ok = maybe_write_image_from_imagedata(data, img_path)
            if not ok:
                warnings.append(f"[WARN] 找不到對應影像（亦無 imageData）：{jpath}")
                continue

        shapes = data.get("shapes", [])
        lines: List[str] = []
        for sh in shapes:
            label = sh.get("label")
            if label not in label_to_id:
                warnings.append(f"[WARN] label '{label}' 不在 names.txt，略過（{jpath.name}）")
                continue
            poly = shape_to_polygon(sh)
            if not poly:
                # 不支援的 shape 或點數不足
                continue
            poly = remove_near_duplicate_points(poly, tol=0.5)
            if len(poly) < 3:
                continue
            poly01 = normalize_polygon(poly, iw, ih)
            line = polygon_to_yolo_line(label_to_id[label], poly01)
            if line:
                lines.append(line)

        if not lines and skip_empty_label:
            # 不產生空標註；略過此影像
            continue

        items.append((img_path, lines))

    return items, warnings


def split_items(n: int, val_ratio: float, test_ratio: float, seed: int = 42) -> Dict[str, List[int]]:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_test = int(round(n * test_ratio))
    n_val = int(round(n * val_ratio))
    n_train = max(0, n - n_val - n_test)
    # 保證總數相等：優先調整 val，再 test
    if n_train + n_val + n_test != n:
        diff = n - (n_train + n_val + n_test)
        n_val = max(0, n_val + diff)
        if n_train + n_val + n_test != n:
            diff = n - (n_train + n_val + n_test)
            n_test = max(0, n_test + diff)

    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val: n_train + n_val + n_test],
    }
    return splits


def write_dataset(
    items: List[Tuple[Path, List[str]]],
    out_dir: Path,
    splits: Dict[str, List[int]],
    force_copy: bool,
    create_empty_label: bool
) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    for split, id_list in splits.items():
        if split == "test" and len(id_list) == 0:
            continue
        img_dir = out_dir / "images" / split
        lab_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        for i in id_list:
            img_path, lines = items[i]
            dst_img = img_dir / img_path.name
            symlink_or_copy(img_path, dst_img, force_copy=force_copy)

            dst_lab = lab_dir / (img_path.stem + ".txt")
            if lines:
                with dst_lab.open("w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line + "\n")
            else:
                if create_empty_label:
                    dst_lab.touch()
            counts[split] += 1

    return counts


def main():
    parser = argparse.ArgumentParser(description="Convert Labelme JSON to Ultralytics YOLO (segmentation) dataset.")
    parser.add_argument("--input", required=True, type=Path, help="Labelme JSON 與影像所在資料夾（可含子資料夾）。")
    parser.add_argument("--output", required=True, type=Path, help="輸出資料集根目錄。")
    parser.add_argument("--names", required=True, type=Path, help="names.txt 路徑（每行一類別）。")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="驗證集比例（0~1）。")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="測試集比例（0~1）。")
    parser.add_argument("--seed", type=int, default=42, help="資料切分隨機種子。")
    parser.add_argument("--copy", action="store_true", help="改為複製檔案（預設嘗試 symlink，失敗再複製）。")
    parser.add_argument("--img-exts", default=".jpg,.jpeg,.png,.bmp", help="搜尋影像的副檔名清單（逗號分隔）。")
    parser.add_argument("--skip-empty-label", action="store_true",
                        help="若影像沒有任何標註則略過（預設：會產生空的 .txt 方便訓練）。")

    args = parser.parse_args()

    input_dir: Path = args.input.resolve()
    out_dir: Path = args.output.resolve()
    names_path: Path = args.names.resolve()
    val_ratio: float = args.val_ratio
    test_ratio: float = args.test_ratio
    seed: int = args.seed
    force_copy: bool = args.copy
    img_exts = [e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
                for e in args.img_exts.split(",") if e.strip()]
    skip_empty = args.skip_empty_label

    if not input_dir.exists():
        print(f"[ERR] input 不存在：{input_dir}", file=sys.stderr)
        sys.exit(1)
    if not names_path.exists():
        print(f"[ERR] names.txt 不存在：{names_path}", file=sys.stderr)
        sys.exit(1)
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0) or (val_ratio + test_ratio >= 1.0):
        print("[ERR] val-ratio 與 test-ratio 需在 [0,1)，且兩者和需 < 1", file=sys.stderr)
        sys.exit(1)

    names = read_names(names_path)
    print(f"[INFO] 讀入 {len(names)} 個類別：{names}")

    print("[INFO] 掃描並轉換 JSON 中的多邊形標註...")
    items, warns = collect_items(input_dir, names, img_exts, skip_empty)
    for w in warns:
        print(w)

    if not items:
        print("[ERR] 找不到可用的標註/影像，或皆被略過。", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 共收集 {len(items)} 張影像。切分 train/val/test ...")
    splits = split_items(len(items), val_ratio, test_ratio, seed=seed)

    # 建立輸出結構與寫入
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = write_dataset(items, out_dir, splits, force_copy=force_copy, create_empty_label=not skip_empty)

    has_test = len(splits.get("test", [])) > 0
    yaml_path = write_data_yaml(out_dir, names, has_test=has_test)

    print("[INFO] 完成！")
    print(f"  - 訓練影像數：{counts['train']}")
    print(f"  - 驗證影像數：{counts['val']}")
    if has_test:
        print(f"  - 測試影像數：{counts['test']}")
    print(f"  - dataset.yaml：{yaml_path}")
    print()
    print("接下來你可以執行（以 Ultralytics 為例）：")
    print(f"  yolo segment train model=yolo12n-seg.yaml data={yaml_path} imgsz=640 epochs=100")
    print()
    print("若你希望保留原始資料夾結構、僅輸出 labels，或是以 COCO 等其他格式中轉，可依需求調整此腳本。")


if __name__ == "__main__":
    main()
