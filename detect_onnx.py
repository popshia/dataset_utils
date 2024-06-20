# Inference using ONNX model
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
    time_synchronized,
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
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        if True in self.video_flag:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
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
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years


def load_video_and_inference(args):
    session_1 = ort.InferenceSession(args.onnx, providers=ORT_PROVIDERS)
    names_1 = get_class_names(args.classes_txt)
    colors_1 = [[random.randint(0, 255) for _ in range(3)] for _ in names_1]

    if args.second_onnx:
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

    t0 = time.time()
    with alive_bar(len(dataset)) as bar:
        human_in_roi = False
        approved = False
        without_buckle = False
        without_buckle_start_time = 0
        nothing = False
        nothing_start_time = 0

        for path, img, im0s, vid_cap, ratio, dwdh in dataset:
            model_predictions = []
            t1 = time_synchronized()

            pred = inference_with_onnx_session(session_1, img)
            model_predictions.append(pred)

            if args.second_onnx:
                pred_2 = inference_with_onnx_session(session_2, img)
                model_predictions.append(pred_2)

            t2 = time_synchronized()

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

            # Process detections
            for i, pred in enumerate(model_predictions):
                names, colors, conf_thres = (
                    (names_1, colors_1, args.conf_thres)
                    if i == 0
                    else (names_2, colors_2, args.second_conf_thres)
                )

                if len(pred):
                    # Print results
                    for c in np.unique(pred[:, -2]):
                        n = np.logical_and(
                            pred[:, -2] == c, pred[:, -1] > conf_thres
                        ).sum()  # detections per class
                        if n > 0:
                            s += (
                                f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            )

                    if args.save_boxes:
                        for i, (_, *xyxy, _, _) in enumerate(pred):  # detection boxes
                            xyxy -= np.array(dwdh * 2)
                            xyxy /= ratio
                            xyxy = xyxy.round().astype(np.int32).tolist()
                            cv2.imwrite(
                                save_path[:-4] + f"_box_{i+1}" + save_path[-4:],
                                im0[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]].copy(),
                            )

                    for _, *xyxy, cls, conf in pred:  # detections per image
                        # if conf > conf_thres:
                        if conf > conf_thres and int(cls) == 0:
                            xyxy -= np.array(dwdh * 2)
                            xyxy /= ratio
                            xyxy = xyxy.round().astype(np.int32).tolist()

                            # check person x coordinate
                            if i == 1 and int(cls) == 0:
                                human_in_roi = (
                                    True
                                    if 765 < (xyxy[0] + xyxy[2]) / 2 < 1470
                                    else False
                                )
                                print(human_in_roi)

                            # Plot box
                            if args.no_label:
                                label = None
                            else:
                                label = f"{names[int(cls)]} {conf:.2f}"

                            plot_one_box(xyxy, im0, label, colors[int(cls)])

                            if args.save_txt:
                                xywh = xyxy2xywh(xyxy / gn)
                                line = (
                                    (cls, *xywh, conf)
                                    if args.save_conf
                                    else (cls, *xywh)
                                )  # label format
                                with open(txt_path + ".txt", "a") as f:
                                    f.write(("%g " * len(line)).rstrip() % line + "\n")

            result_str = (
                f"{s}({(1E3 * (t2 - t1)):.1f}ms) Inference, {int(1/(t2-t1))} fps."
            )
            print(result_str)

            # check human and buckle ############################################################
            # text_pos = (25, 25)
            # red = [0, 0, 255]
            # green = [0, 255, 0]
            # border_img = im0[10 : im0.shape[0] - 10, 10 : im0.shape[1] - 10]
            #
            # if (human_in_roi and "buckle_fastened" not in result_str) or (
            #     human_in_roi
            #     and "person" in result_str
            #     and "buckle_fastened" not in result_str
            # ):
            #     nothing_start_time = 0
            #
            #     if without_buckle_start_time == 0:
            #         without_buckle_start_time = int(time.time())
            #
            #     without_buckle_timer = int(time.time()) - without_buckle_start_time
            #
            #     if 3 <= without_buckle_timer < 6:
            #         text = f"ALARM {without_buckle_timer}"
            #         approved = False
            #     elif without_buckle_timer >= 6:
            #         text = f"SEND ALARM! {without_buckle_timer}"
            #         im0 = cv2.copyMakeBorder(
            #             border_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, red
            #         )
            #         approved = False
            #     else:
            #         text = f"Countdown: {without_buckle_timer}"
            #
            #     if approved:
            #         plot_alarm_message(text_pos, im0, text, green)
            #     else:
            #         plot_alarm_message(text_pos, im0, text, red)
            # elif (
            #     human_in_roi
            #     and "person" in result_str
            #     and "buckle_fastened" in result_str
            # ):
            #     nothing_start_time = 0
            #     without_buckle_start_time = 0
            #     plot_alarm_message(text_pos, im0, "APPROVED!", green)
            #     approved = True
            # else:
            #     if approved:
            #         if nothing_start_time == 0:
            #             nothing_start_time = int(time.time())
            #
            #         nothing_timer = int(time.time()) - nothing_start_time
            #
            #         if nothing_timer < 2:
            #             cv2.putText(
            #                 im0, "APPROVED!", text_pos, 0, 3, green, 3, cv2.LINE_AA
            #             )
            #             plot_alarm_message(text_pos, im0, "APPROVED!", green)
            #         else:
            #             approved = False
            #####################################################################################

            # Stream results
            if args.view_img and (dataset.mode == "video" or dataset.mode == "stream"):
                img_show = (
                    cv2.resize(im0, (1280, 720))
                    if im0.shape[1] > im0.shape[2]
                    else cv2.resize(im0, (720, 1280))
                )
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
    parser.add_argument("--second-classes-txt", type=str, help="second 'classes.txt'")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="conf threshold")
    parser.add_argument(
        "--second-conf-thres", type=float, default=0.5, help="second conf"
    )
    parser.add_argument("--img-size", type=int, default=640, help="model image size")
    parser.add_argument("--project", default="runs/detect", help="save directory")
    parser.add_argument("--name", default="exp", help="current run name")
    parser.add_argument("--no-label", action="store_true", help="don't show label flag")
    parser.add_argument("--save-boxes", action="store_true", help="save boxes flag")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--save-conf", action="store_true", help="save confidences txt")
    parser.add_argument("--exist-ok", action="store_true", help="do not increment flag")
    parser.add_argument("--view-img", action="store_true", help="view result realtime")
    args = parser.parse_args()
    load_video_and_inference(args)
