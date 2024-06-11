# Inference using ONNX model
import argparse
import copy
import random
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import onnxruntime as ort
import torch
from alive_progress import alive_bar

from utils.detect_onnx_utils import clean_str, increment_path, time_synchronized

ORT_PROVIDERS = (
    ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if torch.cuda.is_available()
    else ["CPUExecutionProvider"]
)

IMG_FORMATS = [
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".dng",
    ".webp",
    ".mpo",
]  # acceptable image suffixes

VID_FORMATS = [
    ".mov",
    ".avi",
    ".mp4",
    ".mpg",
    ".mpeg",
    ".m4v",
    ".wmv",
    ".mkv",
]  # acceptable video suffixes


class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        source = Path(path).resolve()  # os-agnostic absolute path
        if source.is_dir():
            files = sorted(source.glob("**/*"))  # dir
        elif source.is_file():
            files = [source]  # files
        else:
            raise Exception(f"ERROR: {source.as_posix()} does not exist")

        images = [
            file.as_posix() for file in files if file.suffix.lower() in IMG_FORMATS
        ]
        videos = [
            file.as_posix() for file in files if file.suffix.lower() in VID_FORMATS
        ]
        image_count, video_count = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.file_count = image_count + video_count  # number of files
        self.video_flag = [False] * image_count + [True] * video_count
        self.mode = "image"
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.file_count > 0, (
            f"No images or videos found in {source}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.file_count:
            raise StopIteration
        path = Path(self.files[self.count]).name

        if self.video_flag[self.count]:
            # Read video
            self.mode = "video"
            ret, frame = self.cap.read()
            if not ret:
                self.count += 1
                self.cap.release()
                if self.count == self.file_count:  # last video
                    raise StopIteration
                else:
                    path = Path(self.files[self.count]).name
                    self.new_video(path)
                    ret, frame = self.cap.read()

            self.frame += 1
            print(
                f"video {self.count + 1}/{self.file_count} ({self.frame}/{self.nframes}) {path}: ",
                end="",
            )

        else:
            # Read image
            self.count += 1
            frame = cv2.imread(path)  # BGR
            assert frame is not None, "Image Not Found " + path
            # print(f'image {self.count}/{self.nf} {path}: ', end='')

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = letter_box(frame, auto=False)
        image, im = proprocess_img(image)

        return path, im, frame, self.cap, ratio, dwdh

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.file_count  # number of files


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, source, img_size=640, stride=32):
        self.mode = "stream"
        self.img_size = img_size
        self.stride = stride

        self.frame = None
        self.sources = [clean_str(x) for x in source]  # clean source names for later
        # Start the thread to read frames from the video stream
        url = eval(source) if source.isnumeric() else source
        cap = cv2.VideoCapture(url)
        assert cap.isOpened(), f"Failed to open {source}"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) % 100

        _, self.frame = cap.read()  # guarantee first frame
        thread = Thread(target=self.update, args=([cap]), daemon=True)
        print(f"{source} success ({w}x{h} at {self.fps:.2f} FPS).")
        thread.start()
        print("")  # newline

    def update(self, cap):
        # Read next stream frame in a daemon thread
        while cap.isOpened():
            # _, self.imgs[index] = cap.read()
            cap.grab()
            success, im = cap.retrieve()
            self.frame = im if success else self.frame * 0
            time.sleep(1 / self.fps)  # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.frame.copy()
        if cv2.waitKey(1) == ord("q"):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img, ratio, dwdh = letter_box(
            cv2.cvtColor(img0, cv2.COLOR_BGR2RGB),
            self.img_size,
            auto=False,
            stride=self.stride,
        )
        image, im = proprocess_img(img)

        return "".join(str(c) for c in self.sources), im, img0, None, ratio, dwdh

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def get_class_names(classes_txt):
    with open(Path(classes_txt).resolve(), "r") as classes:
        cls_names_list = classes.readlines()
        cls_name_list = [cls_name.strip() for cls_name in cls_names_list]
        print("\nClasses: ", cls_name_list, "\n")
        return cls_name_list


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


def plot_one_box(x1, y1, x2, y2, img, ratio, dwdh, color=None, label=None):
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    box = np.array([x1, y1, x2, y2])
    box -= np.array(dwdh * 2)
    box /= ratio
    box = box.round().astype(np.int32).tolist()
    cv2.rectangle(img, box[:2], box[2:], color, thickness=tl, lineType=cv2.LINE_AA)
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
    names = get_class_names(args.classes_txt)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    is_webcam = args.source.isnumeric() or args.source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )

    save_dir = Path(
        increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    )  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)
    vid_path, vid_writer = None, None

    # Set Dataloader
    if is_webcam:
        dataset = LoadStreams(args.source, img_size=args.img_size)
    else:
        dataset = LoadImages(args.source, img_size=args.img_size)

    t0 = time.time()
    with alive_bar(len(dataset)) as bar:
        for path, img, im0s, vid_cap, ratio, dwdh in dataset:
            t1 = time_synchronized()
            pred = inference_with_onnx_session(session, img)
            t2 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if is_webcam:  # batch_size >= 1
                    p, s, im0, _ = str(path), "", im0s, dataset.count
                else:
                    p, s, im0, _ = path, "", im0s, getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                x1, y1, x2, y2, cls, conf = [det[i] for i in range(1, 7)]
                if len(det):
                    # Print results
                    for c in np.unique(pred[:, -2]):
                        n = (pred[:, -2] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Plot box
                    if not args.no_label:
                        label = f"{names[int(cls)]} {conf:.2f}"
                    else:
                        label = None

                    plot_one_box(
                        x1,
                        y1,
                        x2,
                        y2,
                        im0,
                        ratio,
                        dwdh,
                        label=label,
                        color=colors[int(cls)],
                    )

            print(f"{s}({(1E3 * (t2 - t1)):.1f}ms) Inference, {int(1/(t2-t1))} fps.")

            # Stream results
            if args.view_img and (dataset.mode == "video" or dataset.mode == "stream"):
                resized_result = cv2.resize(im0, (1080, 720))
                cv2.imshow(str(p), resized_result)
                if cv2.waitKey(1) == ord("q"):
                    exit(0)

            # Save results (image with detections)
            if dataset.mode == "image":
                cv2.imwrite(save_path, cv2.cvtColor(im0, cv2.COLOR_RGB2BGR))
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += ".mp4"
                    vid_writer = cv2.VideoWriter(
                        save_path, cv2.VideoWriter.fourcc(*"mp4v"), fps, (w, h)
                    )
                vid_writer.write(im0)
            bar()

        print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="onnx_inference.py",
        description="""
Export .pt weight file with command below:
> python export.py --weights ./{YOUR_PT_FILE}.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640

Inference data with yolov7 exported onnx model file.
runs automatically and output frame count and model, bbox infos to stout,
during inference, press "q" to close cv2 window or skip to next data.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("onnx", type=str)
    parser.add_argument("source", type=str)
    parser.add_argument("classes_txt", type=str, help="'classes.txt' file path")
    parser.add_argument("--second-onnx", type=str, help="second model onnx file")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="conf threshold")
    parser.add_argument("--img-size", type=int, default=640, help="model image size")
    parser.add_argument("--project", default="runs/detect", help="save directory")
    parser.add_argument("--name", default="exp", help="current run name")
    parser.add_argument("--no-label", action="store_true", help="don't show label flag")
    parser.add_argument("--save-boxes", action="store_true", help="save boxes flag")
    parser.add_argument("--exist-ok", action="store_true", help="do not increment flag")
    parser.add_argument("--view-img", action="store_true", help="view result realtime")
    args = parser.parse_args()
    load_video_and_inference(args)
    if args.second_onnx:
        second_args = copy.deepcopy(args)
        second_args.onnx = second_args.second_onnx
        load_video_and_inference(args)
