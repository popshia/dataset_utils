## YOLO Dataset Utilities
This is a repo contain multiple scripts that is commonly used by me while processing yolo format annotated dataset before entering the training phase.

All script are placed in `utils`, with an entry point `main.py` in the root.

1. `convert_color.py` - process the dataset with four other color conversion to achieve data augmentation.
```bash
usage: convert_color.py [-h] dataset_dir

positional arguments:
  dataset_dir  dataset directory
```
2. `count_object.py` - count the objects in the whole dataset.
```bash
usage: count_object.py [-h] dataset_dir

positional arguments:
  dataset_dir  dataset directory
```
3. `detect_onnx.py` - inference videos or rtsp stream with `.onnx` file using onnx runtime.
> First export yolov7 `.pt` weight file with command below:
```bash
python export.py --weights ./{YOUR_PT_FILE}.pt \
                 --grid --end2end --simplify --topk-all 100 \
                 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
```
```bash
usage: onnx_inference.py [-h] [--save-video] [--no-label]
                         onnx input img_size conf_thr classes_txt


Inference data with yolov7 exported onnx model file.
runs automatically and output frame count and model, bbox infos to stout,
during inference, press "q" to close cv2 window or skip to next data.

positional arguments:
  onnx          onnx file
  input         inference rtsp url, single file or video directory
  img_size      img size used in training phase
  conf_thr      conf threshold
  classes_txt   'classes.txt' file path

optional arguments:
  --save-video  save detection output video
  --no-label    do not plot label in the results
```
4. `detect_onnx_img.py` - inference image files with onnx file.
```bash
usage: detect_onnx_img.py [-h] onnx input

positional arguments:
  onnx        onnx file
  input       inference img directory
```
5. `draw_yolo_box.py` - plot yolo labels on images and save to `./save_imgs`
```bash
usage: draw_yolo_boxes.py [-h] dataset_dir classes_txt output_count

positional arguments:
  dataset_dir   dataset directory
  classes_txt   'classes.txt' file path
  output_count  the amount of images to plot
```
6. `generate_txt.py` - generate yolo's `train.txt` and `val.txt`.
```bash
usage: generate_txt.py [-h] [--aug] train_dir val_dir

positional arguments:
  train_dir   training dataset directory
  val_dir     validation dataset directory
```
7. `label_format_converter.py` - a revised version of the GREATEST jack_lin's original script, only support xml to txt at this moment.
```bash
usage: label_format_converter.py [-h]
                                 input_format output_format label_dir
                                 classes_txt

positional arguments:
  input_format   input label format
  output_format  output label format
  label_dir      input label directory
  classes_txt    'classes.txt' file path
```
8. `split_val.py` - split dataset as train/val split with user input percentage.
```bash
usage: split_val.py [-h] dataset_dir train_percentage

positional arguments:
  dataset_dir       dataset directory
  train_percentage  precentange of train split (1-100)
```
