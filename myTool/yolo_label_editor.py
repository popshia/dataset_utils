import os
import argparse

def process_annotation_lines(lines, src_class: str, dst_class: str = None):
    """
    處理一個檔案的所有行，回傳新的行列表：
    - 如果只給 src_class，則刪除所有以 src_class 開頭的行
    - 如果還給了 dst_class，則把所有以 src_class 開頭的行，替換成以 dst_class 開頭
    """
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts and parts[0] == src_class:
            if dst_class is None:
                # 刪除這一行
                continue
            else:
                # 將 class id 改成 dst_class
                parts[0] = dst_class
                line = ' '.join(parts) + '\n'
        new_lines.append(line)
    return new_lines

def batch_process_folder(folder_path: str, src_class: str, dst_class: str = None,
                         ext: str = '.txt', output_folder: str = None) -> None:
    """
    遞迴掃描資料夾，處理所有指定副檔名的檔案，
    如果提供 output_folder，則把結果寫到對應的相對路徑下；否則覆蓋原檔。
    """
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if not fname.lower().endswith(ext):
                continue
            in_path = os.path.join(root, fname)
            # 讀入原始檔案
            with open(in_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            new_lines = process_annotation_lines(lines, src_class, dst_class)

            # 決定輸出路徑
            if output_folder:
                rel_dir = os.path.relpath(root, folder_path)
                out_dir = os.path.join(output_folder, rel_dir)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, fname)
            else:
                out_path = in_path

            # 寫出結果
            with open(out_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)

            action = f"{src_class} 改成 {dst_class}" if dst_class else f"已刪除 {src_class}"
            print(f"[{action}] {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="刪除或轉換 YOLO 標註檔中的類別編號，並可選擇輸出到新資料夾")
    parser.add_argument('-f', '--folder', required=True,
                        help="要處理的資料夾路徑")
    parser.add_argument('-c', '--class', dest='src_class', required=True,
                        help="原本的類別編號 (例如 '8')")
    parser.add_argument('-t', '--to-class', dest='dst_class',
                        help="要轉換成的類別編號 (例如 '5')，不指定則刪除 src_class")
    parser.add_argument('-e', '--ext', default='.txt',
                        help="要處理的檔案副檔名，預設為 .txt")
    parser.add_argument('-o', '--output-folder', dest='output_folder',
                        help="輸出資料夾路徑，若指定則不覆蓋原始檔案")

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
        output_folder=args.output_folder
    )
    print("全部處理完成。")

if __name__ == '__main__':
    main()
