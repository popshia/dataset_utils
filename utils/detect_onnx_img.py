import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

ONNX_FILE = "./0516.onnx"

ORT_PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if torch.cuda.is_available()
    else ["CPUExecutionProvider"]
)
SESSION = ort.InferenceSession(ONNX_FILE, providers=ORT_PROVIDERS)
CLASSES = [
    "H61-300621",
    "H69-300690",
    "H69-300641",
    "H69-400851",
    "H61-401052",
    "H69-300590",
    "H69-300850",
    "H69-300650",
]
COLORS = {
    name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(CLASSES)
}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="inference img directory")
    args = parser.parse_args()
    for img in Path(args.input).resolve().glob("**/*[JP][PN]G"):
        print(img.as_posix())
        input_img = cv2.imread(img.as_posix())
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        image = input_img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        outname = [i.name for i in SESSION.get_outputs()]
        inname = [i.name for i in SESSION.get_inputs()]
        inp = {inname[0]: im}

        # ONNX inference
        outputs = SESSION.run(outname, inp)[0]
        ori_images = [input_img.copy()]

        for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
            image = ori_images[int(batch_id)]
            box = np.array([x0, y0, x1, y1])
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score), 3)
            name = CLASSES[cls_id]
            color = COLORS[name]
            name += " " + str(score)
            tl = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
            tf = max(tl - 1, 1)
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

        cv2.imwrite("./demo_imgs/" + img.name, image)
