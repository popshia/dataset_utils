import os

# 請將 folder_rpath 設定為你要處理的資料夾路徑
folder_path = r"F:\20251014-filtered\12"

# 遍歷資料夾中的所有檔案
for filename in os.listdir(folder_path):
    # 如果檔案是 XML 檔案
    if filename.lower().endswith('.xml'):
        # 取得不含副檔名的檔名（基礎檔名）
        base_name = os.path.splitext(filename)[0]
        # 對應的 JPG 檔案名稱
        jpg_filename = base_name + ".jpg"
        # 完整路徑
        jpg_full_path = os.path.join(folder_path, jpg_filename)
        
        # 檢查相對應的 JPG 檔案是否存在
        if not os.path.exists(jpg_full_path):
            xml_full_path = os.path.join(folder_path, filename)
            try:
                os.remove(xml_full_path)
                print(f"已刪除：{xml_full_path}")
            except Exception as e:
                print(f"刪除 {xml_full_path} 時發生錯誤：{e}")
