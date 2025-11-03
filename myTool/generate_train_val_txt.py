import argparse
import random
from pathlib import Path

def collect_files(dir_paths, exts):
    """
    遞迴收集所有指定資料夾內符合副檔名的檔案（去重、回傳絕對路徑）。
    """
    files = set()
    for dir_path in dir_paths:
        p = Path(dir_path)
        if not p.exists():
            print(f"警告：目錄 {dir_path} 不存在，將略過此路徑！")
            continue
        for file_path in p.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in exts:
                    files.add(str(file_path.resolve()))
    return list(files)

def split_files_three(file_list, train_ratio, val_ratio, test_ratio, ensure_nonempty=False):
    """
    隨機三分割檔案清單；可選擇確保 train/val 非空（在檔案數允許時），test 可為 0。
    """
    n = len(file_list)
    shuffled = file_list[:]  # 避免原清單被改動
    random.shuffle(shuffled)

    # 先用地板分配，最後一組用殘差吃掉以避免加總誤差
    train_count = int(n * train_ratio)
    val_count = int(n * val_ratio)
    test_count = n - train_count - val_count

    if ensure_nonempty and n >= 2:
        # 僅強制 train/val 至少 1（若其比例 > 0 且目前為 0）
        if train_ratio > 0 and train_count == 0:
            # 從較大的分區借 1（優先 test，再 val）
            if test_count > 0:
                test_count -= 1
                train_count += 1
            elif val_count > 1:
                val_count -= 1
                train_count += 1
        if val_ratio > 0 and val_count == 0:
            if test_count > 0:
                test_count -= 1
                val_count += 1
            elif train_count > 1:
                train_count -= 1
                val_count += 1
        # 不強制 test 非空；test 可以為 0

    train_files = shuffled[:train_count]
    val_files = shuffled[train_count:train_count + val_count]
    test_files = shuffled[train_count + val_count:]
    return train_files, val_files, test_files

def write_list_to_file(file_list, filename, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        for item in file_list:
            f.write(item + "\n")
    print(f"{output_path} 已成功產生（{len(file_list)} 筆）。")

def resolve_ratios(train_ratio, val_ratio, test_ratio):
    """
    解析三個比例的關係：
    - 允許 val 或 test 其中之一為 None（自動補齊剩餘比例）
    - 若三者皆提供，需和為 1
    - 若 val 與 test 皆為 None，預設 test=0、val=1-train
    """
    eps = 1e-9
    # 邊界檢查
    for name, v in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if v is not None and not (0.0 - eps <= v <= 1.0 + eps):
            raise ValueError(f"{name} 必須介於 0 與 1 之間。")

    # 三者皆提供
    if val_ratio is not None and test_ratio is not None:
        s = train_ratio + val_ratio + test_ratio
        if abs(s - 1.0) > 1e-6:
            raise ValueError("當 train_ratio、val_ratio 與 test_ratio 都有提供時，三者加總必須為 1。")
        return train_ratio, val_ratio, test_ratio

    # val 缺、test 有 -> val = 1 - train - test
    if val_ratio is None and test_ratio is not None:
        v = 1.0 - train_ratio - test_ratio
        if v < -1e-6:
            raise ValueError("train_ratio + test_ratio 不可超過 1。")
        return train_ratio, max(0.0, v), test_ratio

    # val 有、test 缺 -> test = 1 - train - val
    if val_ratio is not None and test_ratio is None:
        t = 1.0 - train_ratio - val_ratio
        if t < -1e-6:
            raise ValueError("train_ratio + val_ratio 不可超過 1。")
        return train_ratio, val_ratio, max(0.0, t)

    # val/test 皆缺 -> 預設 test=0、val=1-train
    v = 1.0 - train_ratio
    if v < -1e-6:
        raise ValueError("train_ratio 不可超過 1。")
    return train_ratio, max(0.0, v), 0.0

def main():
    parser = argparse.ArgumentParser(description="根據指定資料夾產生 train.txt / val.txt / test.txt")
    parser.add_argument("folders", nargs="+", help="請指定一個或多個資料夾路徑")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="訓練集比例（預設 0.8）")
    parser.add_argument("--val_ratio", type=float, default=None, help="驗證集比例（可省略，將由剩餘比例推算）")
    parser.add_argument("--test_ratio", type=float, default=None, help="測試集比例（可省略，預設為 0；亦可明確指定為 0）")
    parser.add_argument("--output_dir", type=str, default=".", help="指定輸出目錄，預設為目前工作目錄")
    parser.add_argument("--ext", nargs="+", default=[".jpg", ".jpeg", ".png"], help="要收集的副檔名（含點，預設: .jpg .jpeg .png）")
    parser.add_argument("--seed", type=int, default=None, help="隨機種子（指定以獲得可重現分割）")
    parser.add_argument("--ensure_nonempty", action="store_true", help="確保 train/val 兩側皆非空（在樣本數允許時）")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    exts = {e.lower() if e.startswith(".") else "." + e.lower() for e in args.ext}

    all_files = collect_files(args.folders, exts)
    if not all_files:
        print("找不到任何檔案，請檢查指定的資料夾路徑與副檔名。")
        return

    # 解析與驗證比例
    train_ratio, val_ratio, test_ratio = resolve_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    train_files, val_files, test_files = split_files_three(
        all_files, train_ratio, val_ratio, test_ratio, args.ensure_nonempty
    )

    write_list_to_file(train_files, "train.txt", args.output_dir)
    write_list_to_file(val_files, "val.txt", args.output_dir)
    write_list_to_file(test_files, "test.txt", args.output_dir)

    print(
        f"總計 {len(all_files)} 筆；"
        f"train: {len(train_files)}，val: {len(val_files)}，test: {len(test_files)}；"
        f"實際比例約為 "
        f"train={len(train_files)/len(all_files):.3f}, "
        f"val={len(val_files)/len(all_files):.3f}, "
        f"test={len(test_files)/len(all_files):.3f}"
    )

if __name__ == "__main__":
    main()
