import os
import cv2
import sys

def expand_bbox(folder, extension_list=['.jpg', '.png', '.jpeg'], delta=5):
    # 遍歷資料夾中所有檔案
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            txt_path = os.path.join(folder, filename)
            base = os.path.splitext(filename)[0]
            
            # 根據檔名尋找對應的影像檔
            image_file = None
            for ext in extension_list:
                candidate = os.path.join(folder, base + ext)
                if os.path.exists(candidate):
                    image_file = candidate
                    break
            if image_file is None:
                print(f"Warning: 找不到 {base} 的圖像文件.")
                continue
            
            # 讀取影像取得寬度與高度
            img = cv2.imread(image_file)
            if img is None:
                print(f"Warning: 無法讀取 {image_file}.")
                continue
            h_img, w_img = img.shape[:2]
            
            # 讀取標註檔，處理每一行
            new_lines = []
            with open(txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"Warning: {txt_path} 異常行： {line}")
                    new_lines.append(line)
                    continue
                cls, x_center_norm, y_center_norm, w_norm, h_norm = parts
                x_center_norm = float(x_center_norm)
                y_center_norm = float(y_center_norm)
                w_norm = float(w_norm)
                h_norm = float(h_norm)
                
                # 將 normalized 座標轉換成 pixel 座標
                x1 = (x_center_norm - w_norm/2) * w_img
                y1 = (y_center_norm - h_norm/2) * h_img
                x2 = (x_center_norm + w_norm/2) * w_img
                y2 = (y_center_norm + h_norm/2) * h_img
                
                # 擴展 bounding box，左右上下各增加 delta 像素
                x1_new = max(0, x1 - delta)
                y1_new = max(0, y1 - delta)
                x2_new = min(w_img, x2 + delta)
                y2_new = min(h_img, y2 + delta)
                
                # 轉換回 normalized 座標：計算中心與寬高
                new_x_center_norm = ((x1_new + x2_new) / 2) / w_img
                new_y_center_norm = ((y1_new + y2_new) / 2) / h_img
                new_w_norm = (x2_new - x1_new) / w_img
                new_h_norm = (y2_new - y1_new) / h_img
                
                new_line = f"{cls} {new_x_center_norm:.6f} {new_y_center_norm:.6f} {new_w_norm:.6f} {new_h_norm:.6f}\n"
                new_lines.append(new_line)
            
            # 將更新後的標註寫回原檔案
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            print(f"已處理： {txt_path}，對應影像： {image_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python expand_bbox.py 資料夾路徑")
        sys.exit(1)
    folder_path = sys.argv[1]
    expand_bbox(folder_path)
