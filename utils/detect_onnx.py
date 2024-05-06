# Inference using ONNX model
import argparse
import os
import pprint
import random
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

ORT_PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if torch.cuda.is_available()
    else ["CPUExecutionProvider"]
)


def get_class_names(classes_txt):
    with open(Path(classes_txt).resolve(), "r") as classes:
        cls_names_list = classes.readlines()
        cls_name_list = [cls_name.strip() for cls_name in cls_names_list]
        print("\nClasses: ", cls_name_list, "\n")
        return cls_name_list


def set_color_by_class_count(classes):
    return {
        name: [random.randint(0, 255) for _ in range(3)]
        for _, name in enumerate(classes)
    }


def proprocess_img(image):
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    nor_image = image.astype(np.float32)
    nor_image /= 255
    return image, nor_image


def inference_with_onnx_session(session, im):
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    input = {inname[0]: im}
    model_outs = session.run(outname, input)[0]
    pprint.pprint(model_outs)
    return model_outs


def plot_and_show_results(outputs, org_imgs, dwdh, ratio, conf_thr, classes, colors):
    detect_results = []

    for _, (batch_id, x1, y1, x2, y2, cls_id, conf) in enumerate(outputs):
        if conf >= conf_thr:
            # read img
            image = org_imgs[int(batch_id)]
            # set box thickness
            tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
            # coordinates
            box = np.array([x1, y1, x2, y2])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            # cls_id and cls_name
            cls_id = int(cls_id)
            label = classes[cls_id]
            color = colors[label]
            label += " " + str(conf)
            # conf
            conf = round(float(conf), 3)
            # get detections
            detection = image[box[1] : box[3], box[0] : box[2]]
            # cv2.imshow("Result", detection)
            # draw box
            cv2.rectangle(
                image, box[:2], box[2:], color, thickness=tl, lineType=cv2.LINE_AA
            )
            detect_results.append(detection)
            # label box
            c1, c2 = box[:2], box[2:]
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
            cv2.putText(
                image,
                label,
                (c1[0], c1[1] - 2),
                0,
                tl / 3,
                [255, 255, 255],
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    resized_result = cv2.resize(
        cv2.cvtColor(org_imgs[0], cv2.COLOR_RGB2BGR), (1080, 720)
    )
    cv2.imshow("Result", resized_result)
    return detect_results


def letter_box(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)


def load_video_and_inference(args):
    session = ort.InferenceSession(args.onnx, providers=ORT_PROVIDERS)
    classes = get_class_names(args.classes_txt)
    colors = set_color_by_class_count(classes)

    if args.input[:4] == "rtsp" or not Path(args.input).is_dir():
        print("\n", end="")
        print(args.input)
        print("-" * os.get_terminal_size().columns)
        if "rtsp://" in args.input:
            cap = cv2.VideoCapture(args.input)
        else:
            cap = cv2.VideoCapture(Path(args.input).resolve().as_posix())
        frame_count = 1

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame. Exiting ...")
                break

            start = time.time()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            org_imgs = [frame.copy()]
            image = frame.copy()
            image, ratio, dwdh = letter_box(image, auto=False)
            image, im = proprocess_img(image)
            model_outputs = inference_with_onnx_session(session, im)
            detection_results = plot_and_show_results(
                model_outputs, org_imgs, dwdh, ratio, args.conf_thr, classes, colors
            )

            # inference time of first data's first frame involves loading data onto gpu, ignore due to not accurate
            if frame_count == 1:
                pass
            else:
                print(
                    "frame count:",
                    frame_count,
                    "\ninference time:",
                    time.time() - start,
                    "\nfps:",
                    1 / (time.time() - start),
                )

            if cv2.waitKey(1) == ord("q"):
                break

            frame_count += 1
            print("-" * os.get_terminal_size().columns)

        cap.release()
    else:
        for i, data in enumerate(Path(args.input).resolve().glob("**/*")):
            print("\n", end="")
            print(i + 1, ":", data)
            print("-" * os.get_terminal_size().columns)
            cap = cv2.VideoCapture(data.resolve().as_posix())
            frame_count = 1

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    print("Can't receive frame. Exiting ...")
                    break

                start = time.time()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                org_imgs = [frame.copy()]
                image = frame.copy()
                image, ratio, dwdh = letter_box(
                    image, new_shape=(args.img_size, args.img_size), auto=False
                )
                image, im = proprocess_img(image)
                model_outputs = inference_with_onnx_session(session, im)
                detection_results = plot_and_show_results(
                    model_outputs, org_imgs, dwdh, ratio, args.conf_thr, classes, colors
                )

                # inference time of first data's first frame involves loading data onto gpu, ignore due to not accurate
                if i == 0 and frame_count == 1:
                    pass
                else:
                    print(
                        "frame count:",
                        frame_count,
                        "\ninference time:",
                        time.time() - start,
                        "\nfps:",
                        1 / (time.time() - start),
                    )

                if cv2.waitKey(1) == ord("q"):
                    break

                frame_count += 1
                print("-" * os.get_terminal_size().columns)

            cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="onnx_inference.py",
        description="""
#######################################################
# CHANGE THE CLASS NAMES ON #14 FIRST !!!!!!!!!!!!!!! #
#######################################################

Export .pt weight file with command below:
> python export.py --weights ./{YOUR_PT_FILE}.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
  # For onnxruntime, you need to specify this value as an integer,
  # when it is 0 it means agnostic NMS,
  # otherwise it is non-agnostic NMS

Inference data with yolov7 exported onnx model file.
runs automatically and output frame count and model, bbox infos to stout,
during inference, press "q" to close cv2 window or skip to next data.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("onnx", type=str, help="onnx file")
    parser.add_argument(
        "input", type=str, help="inference rtsp url, single file or video directory"
    )
    parser.add_argument("img_size", type=int, help="img size used in training phase")
    parser.add_argument("conf_thr", type=float, help="conf threshold")
    parser.add_argument("classes_txt", type=str, help="'classes.txt' file path")
    args = parser.parse_args()
    load_video_and_inference(args)
