#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_organizer.py
將標註檔與影像依正/負樣本分類並搬移。
"""

import os
import shutil
import argparse
import xml.etree.ElementTree as ET

def is_negative_sample(xml_path, neg_classes):
    """
    解析 XML，判斷是否為負樣本：
    - 完全沒有 <object>
    - 或所有 <object>/<name> 都在 neg_classes
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    # 空白樣本視為負樣本
    if not objects:
        return True
    # 檢查是否所有名稱都屬於負樣本清單
    for obj in objects:
        name = obj.find('name').text.strip()
        if name not in neg_classes:
            return False
    return True

def organize_dataset(input_dir, neg_classes, pos_dir, neg_dir):
    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    # 掃描所有 XML
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.xml'):
            continue
        xml_path = os.path.join(input_dir, fname)
        base, _ = os.path.splitext(fname)
        jpg_name = base + '.jpg'
        jpg_path = os.path.join(input_dir, jpg_name)

        # 判斷並選擇目標資料夾
        if is_negative_sample(xml_path, neg_classes):
            target_dir = neg_dir
        else:
            target_dir = pos_dir

        # 搬移 XML
        shutil.move(xml_path, os.path.join(target_dir, fname))
        # 如果對應的 JPG 存在，也一併搬移
        if os.path.exists(jpg_path):
            shutil.move(jpg_path, os.path.join(target_dir, jpg_name))

        print(f"已將 {base} 移動到 「{os.path.basename(target_dir)}」 資料夾")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="依正/負樣本分類並搬移資料集中的 XML 與 JPG 檔案")
    parser.add_argument('--input_dir', required=True,
                        help="原始資料夾路徑")
    parser.add_argument('--neg_classes', nargs='+', required=True,
                        help="負樣本類別清單，用空白分隔")
    parser.add_argument('--pos_dir', default='positive',
                        help="正樣本目標資料夾，預設 'positive'")
    parser.add_argument('--neg_dir', default='negative',
                        help="負樣本目標資料夾，預設 'negative'")
    args = parser.parse_args()

    organize_dataset(
        input_dir=args.input_dir,
        neg_classes=set(args.neg_classes),
        pos_dir=args.pos_dir,
        neg_dir=args.neg_dir
    )
    
"""

python dataset_organizer.py `
  --input_dir 'e:\temp\20250425_received_from_David_Danhai_LRT_recording\images' `
  --neg_classes person `
  --pos_dir 'e:\temp\20250425_received_from_David_Danhai_LRT_recording\positive' `
  --neg_dir 'e:\temp\20250425_received_from_David_Danhai_LRT_recording\negative'


"""
