import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path


def load_class_name(classes_txt):
    with open(os.path.abspath(classes_txt), "r") as classes:
        cls_names_list = classes.readlines()
        cls_name_list = [cls_name.strip() for cls_name in cls_names_list]
        print("\nClasses: ", cls_name_list, "\n")
        return cls_name_list


def parse_and_convert_label(label, cls_name_list):
    print("Org file path: ", label)

    # read label
    tree = ET.parse(label)
    root = tree.getroot()

    objs_list = root.findall("object")

    # get image size
    for img_size in root.findall("size"):
        img_w = float(img_size[0].text)
        img_h = float(img_size[1].text)

    new_labels = []

    # get objs info (coordinate, class name)
    for obj in objs_list:
        for obj_name in obj.findall("name"):
            obj_cls_name = obj_name.text

        for obj_bndbox in obj.findall("bndbox"):
            xmin = float(obj_bndbox[0].text)
            ymin = float(obj_bndbox[1].text)
            xmax = float(obj_bndbox[2].text)
            ymax = float(obj_bndbox[3].text)

        cls_id = cls_name_list.index(obj_cls_name)

        # convert label
        out_str = "{} {} {} {} {}\n".format(
            cls_id,
            (xmin + xmax) / (2 * img_w),
            (ymin + ymax) / (2 * img_h),
            (xmax - xmin) / img_w,
            (ymax - ymin) / img_h,
        )
        new_labels.append(out_str)

    return new_labels


def output_new_label(org_label_path, new_label, output_format):
    output_label_file_path = Path(org_label_path).with_suffix("." + output_format)
    print("New label path: ", output_label_file_path)
    print("-" * os.get_terminal_size().columns)
    with open(str(output_label_file_path), "w") as outfile:
        outfile.writelines(new_label)


def convert_label_format(opts):
    input_format = opts.input_format
    output_format = opts.output_format
    input_dir = opts.label_dir
    classes_txt = opts.classes_txt

    cls_name_list = load_class_name(classes_txt)
    label_list = sorted(Path(input_dir).glob("**/*." + input_format))
    for label in label_list:
        new_label = parse_and_convert_label(label, cls_name_list)
        output_new_label(label, new_label, output_format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_format", help="input label format")
    parser.add_argument("output_format", help="output label format")
    parser.add_argument("label_dir", help="input label directory")
    parser.add_argument("classes_txt", help="'classes.txt' file path")
    args = parser.parse_args()
    convert_label_format(args)
