import random

import cv2
import numpy as np


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
    return model_outs


def plot_one_box(box, img, label, color):
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    # plot bounding box
    cv2.rectangle(img, box[:2], box[2:], color, thickness=tl, lineType=cv2.LINE_AA)
    # plot label
    if label:
        c1, c2 = box[:2], box[2:]
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_one_message(org, img, text, color):
    x, y = org
    t_size = cv2.getTextSize(text, 0, fontScale=2, thickness=3)[0]
    t_w, t_h = t_size
    # sum
    cv2.rectangle(
        img,
        (x - 5, y - 5),
        (x + t_w + 5, y + t_h + 5),
        color,
        -1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        (x, y + t_h),
        0,
        2,
        [255, 255, 255],
        thickness=3,
        lineType=cv2.LINE_AA,
    )
