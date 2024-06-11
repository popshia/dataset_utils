import argparse
import random
import threading
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ORT_PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if torch.cuda.is_available()
    else ["CPUExecutionProvider"]
)


def get_class_names(classes_txt):
    with open(Path(classes_txt).resolve(), "r") as classes:
        cls_names_list = classes.readlines()
        cls_name_list = [cls_name.strip() for cls_name in cls_names_list]
        return cls_name_list


def letterbox(
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


def process_img(image):
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255
    return image, im


def inference_with_onnx_session(session, im):
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    input = {inname[0]: im}
    model_outs = session.run(outname, input)[0]
    return model_outs


def detect_img(img, classes_txt, ort_session):
    print(img.as_posix())
    input_img = cv2.imread(img.as_posix())
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    image = input_img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image, im = process_img(image)
    outputs = inference_with_onnx_session(ort_session, im)
    ori_images = [input_img.copy()]

    classes = get_class_names(classes_txt)
    colors = {
        name: [random.randint(0, 255) for _ in range(3)]
        for _, name in enumerate(classes)
    }
    detection_results = []

    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = classes[cls_id]
        color = colors[name]
        name += " " + str(score)
        tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)
        detection = cv2.cvtColor(
            image[box[1] + tl : box[3] - tl, box[0] + tl : box[2] - tl],
            cv2.COLOR_RGB2BGR,
        )
        detection_results.append(detection)
        cv2.rectangle(
            image, box[:2], box[2:], color, thickness=tl, lineType=cv2.LINE_AA
        )
        c1, c2 = box[:2], box[2:]
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(name, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            image,
            name,
            (box[0], box[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    if not Path("detection_results").is_dir():
        Path("detection_results").mkdir()

    cv2.imwrite(
        "./detection_results/" + img.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    )
    return detection_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx", type=str, help="onnx file")
    parser.add_argument("input", type=str, help="inference img directory")
    parser.add_argument("classes", type=str, help="classes.txt file")
    parser.add_argument(
        "--save-boxes", action="store_true", help="save detection_boxes"
    )
    args = parser.parse_args()
    ort_session = ort.InferenceSession(args.onnx, providers=ORT_PROVIDERS)
    img_list = sorted(Path(args.input).glob("**/*.[jJpP][pPnN][gG]"))
    thread_list = []
    result_boxes = []

    if args.save_boxes:
        for img in img_list:
            result_boxes.append([img, detect_img(img, args.classes, ort_session)])

        for result in result_boxes:
            for i, box in enumerate(result[1]):
                cv2.imwrite(
                    "./detection_results/{}_{:d}.jpg".format(result[0].stem, i),
                    box,
                )
    else:
        for img in img_list:
            detect_img(img, args.classes, ort_session)
        #     thread_list.append(
        #         threading.Thread(
        #             target=detect_img, args=(img, args.classes, ort_session)
        #         )
        #     )
        #
        # for thread in thread_list:
        #     thread.start()
