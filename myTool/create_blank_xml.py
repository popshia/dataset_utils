import os
import xml.etree.ElementTree as ET
from PIL import Image

def create_blank_xml(image_path, xml_path):
    """
    根據圖片資訊產生空白標註 xml，內容包含基本資訊（資料夾、檔名、路徑、圖片尺寸）
    """
    # 使用 Pillow 取得圖片尺寸與通道數（RGB 則為 3）
    im = Image.open(image_path)
    width, height = im.size
    depth = len(im.getbands())
    
    # 建立 XML 樹狀結構
    annotation = ET.Element("annotation")
    
    folder_elem = ET.SubElement(annotation, "folder")
    # 取得影像所在的資料夾名稱
    folder_elem.text = os.path.basename(os.path.dirname(image_path))
    
    filename_elem = ET.SubElement(annotation, "filename")
    filename_elem.text = os.path.basename(image_path)
    
    path_elem = ET.SubElement(annotation, "path")
    path_elem.text = os.path.abspath(image_path)
    
    source_elem = ET.SubElement(annotation, "source")
    database_elem = ET.SubElement(source_elem, "database")
    database_elem.text = "Unknown"
    
    size_elem = ET.SubElement(annotation, "size")
    width_elem = ET.SubElement(size_elem, "width")
    width_elem.text = str(width)
    height_elem = ET.SubElement(size_elem, "height")
    height_elem.text = str(height)
    depth_elem = ET.SubElement(size_elem, "depth")
    depth_elem.text = str(depth)
    
    segmented_elem = ET.SubElement(annotation, "segmented")
    segmented_elem.text = "0"
    
    # 這裡不加入 <object> 節點，表示該影像中沒有標註目標

    # 將 XML 寫入檔案，使用 UTF-8 編碼與 XML 宣告
    tree = ET.ElementTree(annotation)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    print(f"已建立空白 xml：{xml_path}")

def process_folder(folder_path):
    """
    處理指定資料夾內所有 jpg 檔案，檢查是否存在對應的 xml 檔案，若無則建立空白 xml。
    """
    for file in os.listdir(folder_path):
        if file.lower().endswith(".jpg"):
            file_base = os.path.splitext(file)[0]
            xml_file = file_base + ".xml"
            xml_path = os.path.join(folder_path, xml_file)
            # 若 xml 檔案不存在，則建立空白 xml
            if not os.path.exists(xml_path):
                image_path = os.path.join(folder_path, file)
                create_blank_xml(image_path, xml_path)

if __name__ == "__main__":
    folder = input("請輸入資料夾路徑：")
    process_folder(folder)
