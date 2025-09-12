import os
import glob
import random

def generate_train_val_txt(folder_path, train_ratio=9, val_ratio=1):
    """
    根據給定的資料夾路徑，以及 train 與 val 比例，產生 train.txt 與 val.txt。
    
    :param folder_path: 包含圖片與標註檔的資料夾路徑 (str)
    :param train_ratio: train 集的比例 (int)
    :param val_ratio: val 集的比例 (int)
    """
    # 搜尋資料夾中所有 .jpg 檔案
    jpg_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    
    # 如果想確保每張圖片都對應到一個標註 .xml，可在這裡加上檢查
    # 例如：
    # jpg_files = [
    #     f for f in jpg_files
    #     if os.path.exists(os.path.splitext(f)[0] + ".xml")
    # ]
    
    # 將圖片清單隨機排列
    random.shuffle(jpg_files)
    
    total = train_ratio + val_ratio
    train_count = int(len(jpg_files) * (train_ratio / total))
    
    train_files = jpg_files[:train_count]
    val_files = jpg_files[train_count:]
    
    # 將圖片絕對路徑寫入 train.txt
    with open("train.txt", "w", encoding="utf-8") as f_train:
        for img in train_files:
            f_train.write(os.path.abspath(img) + "\n")
    
    # 將圖片絕對路徑寫入 val.txt
    with open("val.txt", "w", encoding="utf-8") as f_val:
        for img in val_files:
            f_val.write(os.path.abspath(img) + "\n")
    
    print(f"產生完成！\ntrain.txt: {len(train_files)} 筆\nval.txt: {len(val_files)} 筆")

if __name__ == "__main__":
    # 範例使用方式：
    # 輸入資料夾路徑 (folder_path) 與想要的比例 (train_ratio, val_ratio)
    folder_path_input = "/path/to/your/dataset"
    generate_train_val_txt(folder_path_input, train_ratio=9, val_ratio=1)
