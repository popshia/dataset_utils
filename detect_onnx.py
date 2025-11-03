# Inference using ONNX model (refactored)
import argparse
import random
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import onnxruntime as ort
import torch
from alive_progress import alive_bar

from utils.detect_onnx_utils import (
    inference_with_onnx_session,
    plot_alarm_message,
    plot_one_box,
    proprocess_img,
)
from utils.general import (
    clean_str,
    get_class_names,
    increment_path,
    letter_box,
    xyxy2xywh,
)


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
        self._video_frame_count_map = {}
        # Pre-calc total frames once for progress accuracy and efficiency
        for v in videos:
            cap = cv2.VideoCapture(v)
            self._video_frame_count_map[v] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        self._image_count = image_count
        self._total_video_frames = sum(self._video_frame_count_map.values())
        self._total_frames = self._image_count + self._total_video_frames

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
        path = self.files[self.count]

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
                    path = self.files[self.count]
                    self.new_video(path)
                    ret, frame = self.cap.read()

            self.frame += 1
            print(
                f"video {self.count + 1}/{self.file_count} ({self.frame}/{self.nframes}) {path}: ",
                end="",
            )

        else:
            # Read image
            self.mode = "image"
            self.count += 1
            frame = cv2.imread(path)  # BGR
            assert frame is not None, "Image Not Found " + path
            # print(f'image {self.count}/{self.nf} {path}: ', end='')

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = letter_box(frame, self.img_size, auto=False)
        image, im = proprocess_img(image)

        return path, im, frame, self.cap, ratio, dwdh

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        # Prefer precomputed frame count for accuracy and speed
        self.nframes = self._video_frame_count_map.get(
            path, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        )

    def __len__(self):
        # Return images + all frames across all videos
        return self._total_frames if True in self.video_flag else self.file_count


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
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = letter_box(
            img0,
            self.img_size,
            auto=False,
            stride=self.stride,
        )
        image, im = proprocess_img(image)

        return "".join(str(c) for c in self.sources), im, img0, None, ratio, dwdh

    def __len__(self):
        return 0  # infinite/unknown length for streams


def parse_csv_ints(s):
    if s is None or len(s.strip()) == 0:
        return None
    return set(int(x.strip()) for x in s.split(",") if x.strip() != "")


def parse_roi(s):
    if s is None or len(s.strip()) == 0:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in s.split(","))
        return (x1, y1, x2, y2)
    except Exception:
        raise argparse.ArgumentTypeError(
            "--roi must be four integers like 'x1,y1,x2,y2'"
        )


def center_in_roi(xyxy, roi_rect):
    if roi_rect is None:
        return False
    x1, y1, x2, y2 = roi_rect
    cx = (xyxy[0] + xyxy[2]) / 2
    cy = (xyxy[1] + xyxy[3]) / 2
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


def load_video_and_inference(args):
    # Prepare sessions and class maps
    session_1 = ort.InferenceSession(args.onnx, providers=ORT_PROVIDERS)
    names_1 = get_class_names(args.classes_txt)
    colors_1 = [[random.randint(0, 255) for _ in range(3)] for _ in names_1]

    has_second = args.second_onnx is not None
    if has_second:
        session_2 = ort.InferenceSession(args.second_onnx, providers=ORT_PROVIDERS)
        names_2 = get_class_names(args.second_classes_txt)
        colors_2 = [[random.randint(0, 255) for _ in range(3)] for _ in names_2]

    is_webcam = args.source.isnumeric() or args.source.lower().startswith(
        ("rtsp://", "rtmp://", "http://", "https://")
    )

    save_dir = Path(
        increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    )  # increment run
    (save_dir / "labels" if args.save_txt else save_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir
    vid_path, vid_writer = None, None

    # Set Dataloader
    if is_webcam:
        dataset = LoadStreams(args.source, img_size=args.img_size)
    else:
        dataset = LoadImages(args.source, img_size=args.img_size)

    # --- Config derived from CLI ---
    target_classes = parse_csv_ints(args.target_classes)
    roi_rect = parse_roi(args.roi)
    roi_models = parse_csv_ints(args.roi_models)
    roi_classes = (
        parse_csv_ints(args.roi_classes)
        if args.roi_classes is not None
        else (parse_csv_ints(args.target_classes) if args.target_classes else None)
    )

    t0 = time.time()
    with alive_bar(len(dataset)) as bar:
        # State for downstream logic (kept for backward-compat)
        human_in_roi = False
        approved = False
        without_buckle = False
        without_buckle_start_time = 0
        nothing = False
        nothing_start_time = 0

        for path, img, im0s, vid_cap, ratio, dwdh in dataset:
            model_predictions = []
            t1 = time.time()

            pred = inference_with_onnx_session(session_1, img)
            model_predictions.append((0, pred, names_1, colors_1, args.conf_thres))

            if has_second:
                pred_2 = inference_with_onnx_session(session_2, img)
                model_predictions.append(
                    (1, pred_2, names_2, colors_2, args.second_conf_thres)
                )

            t2 = time.time()

            if is_webcam:  # batch_size >= 1
                p, s, im0, frame = str(path), "", im0s, dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / "labels" / p.stem) + (
                "" if dataset.mode == "image" else f"_{frame}"
            )  # img.txt
            im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
            gn = np.array(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # Process detections for each model's output
            for model_idx, pred, names, colors, conf_thres in model_predictions:
                if len(pred):
                    # Print results summary
                    for c in np.unique(pred[:, -2]):
                        n = np.logical_and(pred[:, -2] == c, pred[:, -1] > conf_thres).sum()
                        if n > 0:
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                    if args.save_boxes:
                        for j, (_, *xyxy, _, _) in enumerate(pred):  # detection boxes
                            xyxy = np.array(xyxy, dtype=np.float32)
                            xyxy -= np.array(dwdh * 2)
                            xyxy /= ratio
                            xyxy = xyxy.round().astype(np.int32).tolist()
                            x1, y1, x2, y2 = xyxy
                            x1, y1 = max(x1, 0), max(y1, 0)
                            x2, y2 = min(x2, im0.shape[1] - 1), min(y2, im0.shape[0] - 1)
                            cv2.imwrite(
                                save_path[:-4] + f"_box_{j+1}" + save_path[-4:],
                                im0[y1:y2, x1:x2].copy(),
                            )

                    # Draw and optional ROI logic
                    for _, *xyxy, cls, conf in pred:
                        if conf <= conf_thres:
                            continue
                        cls_int = int(cls)
                        if (target_classes is not None) and (cls_int not in target_classes):
                            continue

                        xyxy = np.array(xyxy, dtype=np.float32)
                        xyxy -= np.array(dwdh * 2)
                        xyxy /= ratio
                        xyxy = xyxy.round().astype(np.int32).tolist()

                        # ROI check: configurable
                        if roi_rect is not None:
                            check_model = (roi_models is None) or (model_idx in roi_models)
                            check_class = (roi_classes is None) or (cls_int in roi_classes)
                            if check_model and check_class:
                                if center_in_roi(xyxy, roi_rect):
                                    human_in_roi = True
                                if args.print_roi:
                                    print(f"ROI hit (model={model_idx}, cls={cls_int}): {human_in_roi}")

                        # Plot box & label
                        label = None if args.no_label else f"{names[cls_int]} {conf:.2f}"
                        plot_one_box(xyxy, im0, label, colors[cls_int])

                        if args.save_txt:
                            xywh = xyxy2xywh(np.array(xyxy) / gn)
                            line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)
                            with open(txt_path + ".txt", "a") as f:
                                f.write(("%g " * len(line)).rstrip() % line + "\n")

            result_str = f"{s}({(1E3 * (t2 - t1)):.1f}ms) Inference, {int(1/(t2-t1))} fps."
            print(result_str)

            # (Optional) alarm logic kept commented-out
            # ...

            # Stream results
            if args.view_img and (dataset.mode == "video" or dataset.mode == "stream"):
                # Fix orientation check: compare width vs height
                if im0.shape[1] >= im0.shape[0]:
                    img_show = cv2.resize(im0, (1280, 720))
                else:
                    img_show = cv2.resize(im0, (720, 1280))
                # Optionally draw ROI rectangle for debugging
                if roi_rect is not None and args.draw_roi:
                    x1, y1, x2, y2 = roi_rect
                    cv2.rectangle(img_show, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.imshow("result", img_show)
                if cv2.waitKey(1) == ord("q"):
                    exit(0)

            # Save results (image with detections)
            if dataset.mode == "image":
                cv2.imwrite(save_path, im0)
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
                vid_writer.write(cv2.resize(im0, (w, h)))
            bar()

        print(f"Done. ({time.time() - t0:.3f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="onnx_inference.py",
        description="""
Export .pt weight file with command below:

> python export.py --weights ./{YOUR_PT_FILE}.pt --grid --end2end --simplify --topk-all 100 \
  --iou-thres 0.65 --conf-thres 0.35 --img-size {img_size} {img_size} --max-wh {img_size}

Inference data with yolov7 exported onnx model file.
runs automatically and output frame count and result to stdout,
during inference, press "q" to close result window or skip to next data.""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("onnx", type=str)
    parser.add_argument("source", type=str)
    parser.add_argument("classes_txt", type=str)
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--second-onnx", type=str)
    parser.add_argument("--second-classes-txt", type=str)
    parser.add_argument("--second-conf-thres", type=float, default=0.5)
    parser.add_argument("--img-size", type=int, default=640, help="model image size")
    parser.add_argument("--project", default="runs/detect", help="save directory")
    parser.add_argument("--name", default="exp", help="current run name")
    parser.add_argument("--no-label", action="store_true", help="don't show label")
    parser.add_argument("--save-boxes", action="store_true", help="save detected boxes")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidence to *.txt"
    )
    parser.add_argument(
        "--exist-ok", action="store_true", help="do not increment directory"
    )
    parser.add_argument("--view-img", action="store_true", help="view result")

    # --- New configurables ---
    parser.add_argument(
        "--target-classes",
        type=str,
        default=None,
        help="comma-separated class ids to draw/save (default: all classes)",
    )
    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI rectangle in pixels as x1,y1,x2,y2 (default: disabled)",
    )
    parser.add_argument(
        "--roi-models",
        type=str,
        default=None,
        help="comma-separated model indices to apply ROI on (0=first,1=second; default: all if --roi is set)",
    )
    parser.add_argument(
        "--roi-classes",
        type=str,
        default=None,
        help="comma-separated class ids to apply ROI check on (default: same as --target-classes or all)",
    )
    parser.add_argument(
        "--print-roi",
        action="store_true",
        help="print ROI hit flag for debugging",
    )
    parser.add_argument(
        "--draw-roi",
        action="store_true",
        help="draw ROI rectangle when --view-img is enabled",
    )

    args = parser.parse_args()
    load_video_and_inference(args)
