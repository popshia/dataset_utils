import os
import xml.etree.ElementTree as ET

def remove_empty_annotations(folder_path):
    """
    檢查指定資料夾內的所有 XML 標註檔，若裡面沒有 <object> 則刪除對應的 .xml 與 .jpg
    :param folder_path: 要檢查的資料夾路徑
    """
    # 取得該資料夾內所有檔案清單
    files = os.listdir(folder_path)
    
    # 過濾出 XML 檔案
    xml_files = [f for f in files if f.lower().endswith('.xml')]

    for xml_file in xml_files:
        xml_path = os.path.join(folder_path, xml_file)

        # 解析 XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # 尋找所有 <object> 標籤
        objects = root.findall('object')
        
        # 若沒有任何 <object> 標籤，表示標記為空
        if len(objects) == 0:
            # 找到對應的 jpg 檔案名稱
            # 通常 XML 與 JPG 檔名相同 (副檔名不同)
            # 例如 001.xml 與 001.jpg
            jpg_filename = os.path.splitext(xml_file)[0] + '.jpg'
            jpg_path = os.path.join(folder_path, jpg_filename)
            
            # 刪除 JPG
            if os.path.exists(jpg_path):
                os.remove(jpg_path)
                print(f"已刪除空標記對應的圖片: {jpg_path}")

            # 刪除 XML
            os.remove(xml_path)
            print(f"已刪除空標記的 XML 檔案: {xml_path}")

if __name__ == "__main__":
    # 使用範例:
    # 將 "path_to_folder" 替換為實際存放 XML 與 JPG 的資料夾路徑
    path_to_folder = r"E:\temp\LILIN輕軌標註\20250411\20250407"  # 請改成你的路徑
    remove_empty_annotations(path_to_folder)
