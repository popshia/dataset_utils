# YOLO Dataset Utilities

A comprehensive collection of utilities for processing YOLO format annotated datasets before and after the training phase. These scripts help with data preparation, augmentation, visualization, and analysis of YOLO format datasets.

## Installation

Install all the required dependencies with:

```bash
pip install -r requirements.txt
```

## Scripts Overview

### Data Preparation

| Script | Description |
|--------|-------------|
| [add_empty_labels.py](#add_empty_labelspy) | Creates empty label files for images without annotations |
| [generate_txt.py](#generate_txtpy) | Generates train.txt, val.txt, and test.txt files for YOLO training |
| [label_format_converter.py](#label_format_converterpy) | Converts between different label formats (e.g., XML to YOLO) |
| [split_val.py](#split_valpy) | Splits a dataset into training, validation, and test sets |
| [tweak_label.py](#tweak_labelpy) | Modifies label classes according to predefined mappings |

### Data Augmentation

| Script | Description |
|--------|-------------|
| [bbox_augmentation.py](#bbox_augmentationpy) | Applies various augmentations to images while preserving bounding box annotations |
| [convert_color.py](#convert_colorpy) | Creates color variations of images (brightness, saturation, contrast, invert) |
| [flip_photo.py](#flip_photopy) | Creates flipped versions of images (horizontal, vertical, both) |
| [seg_augmentation.py](#seg_augmentationpy) | Applies augmentations to segmentation datasets |

### Visualization and Analysis

| Script | Description |
|--------|-------------|
| [count_object.py](#count_objectpy) | Counts the number of objects in a dataset |
| [detect_onnx.py](#detect_onnxpy) | Performs inference using ONNX models on images or videos |
| [plot_yolo_labels.py](#plot_yolo_labelspy) | Visualizes YOLO format labels on images |

### Utilities

| Script | Description |
|--------|-------------|
| [capture_frame.py](#capture_framepy) | Captures frames from videos for dataset creation |
| [resize_img.py](#resize_imgpy) | Resizes images to a specified size |

## Detailed Usage

### add_empty_labels.py

Creates empty label files (.txt) for images that don't have corresponding label files.

```bash
python add_empty_labels.py <dataset>
```

**Arguments:**
- `dataset`: Path to the dataset directory containing images

**Example:**
```bash
python add_empty_labels.py /path/to/dataset
```

### bbox_augmentation.py

Applies various augmentations to images while preserving bounding box annotations.

```bash
python bbox_augmentation.py <hyp> <dataset> [--new-image <count>]
```

**Arguments:**
- `hyp`: Path to the hyperparameters YAML file (e.g., hyp.yaml)
- `dataset`: Path to the dataset directory
- `--new-image`: Number of augmented images to generate per original image (default: 5)

**Example:**
```bash
python bbox_augmentation.py hyp.yaml /path/to/dataset --new-image 3
```

### capture_frame.py

Captures frames from videos for dataset creation. Allows interactive frame selection by pressing 's' to save a frame.

```bash
python capture_frame.py <input_dir>
```

**Arguments:**
- `input_dir`: Directory containing video files

**Example:**
```bash
python capture_frame.py /path/to/videos
```

**Controls:**
- Press 's' to save the current frame
- Press 'q' to quit the current video

### convert_color.py

Creates color variations of images (brightness, saturation, contrast, invert) while preserving annotations.

```bash
python convert_color.py <dataset_dir>
```

**Arguments:**
- `dataset_dir`: Path to the dataset directory

**Example:**
```bash
python convert_color.py /path/to/dataset
```

### count_object.py

Counts the total number of objects in a dataset by analyzing label files.

```bash
python count_object.py <dataset_dir>
```

**Arguments:**
- `dataset_dir`: Path to the dataset directory

**Example:**
```bash
python count_object.py /path/to/dataset
```

### detect_onnx.py

Performs inference using ONNX models on images or videos.

```bash
python detect_onnx.py <onnx> <source> <classes_txt> [options]
```

**Arguments:**
- `onnx`: Path to the ONNX model file
- `source`: Path to the source image, video, or directory
- `classes_txt`: Path to the classes.txt file
- `--conf-thres`: Confidence threshold (default: 0.5)
- `--second-onnx`: Path to a second ONNX model for ensemble detection
- `--second-classes-txt`: Path to classes.txt for the second model
- `--second-conf-thres`: Confidence threshold for the second model (default: 0.5)
- `--img-size`: Input image size for the model (default: 640)
- `--project`: Save results to project/name (default: "runs/detect")
- `--name`: Project name (default: "exp")
- `--no-label`: Don't show labels on detections
- `--save-boxes`: Save detected boxes
- `--save-txt`: Save results to *.txt
- `--save-conf`: Save confidence in the text file
- `--exist-ok`: Don't increment the run directory
- `--view-img`: Display results

**Example:**
```bash
python detect_onnx.py model.onnx /path/to/images classes.txt --conf-thres 0.4 --view-img
```

### flip_photo.py

Creates flipped versions of images (horizontal, vertical, both).

```bash
python flip_photo.py <dataset>
```

**Arguments:**
- `dataset`: Path to the dataset directory

**Example:**
```bash
python flip_photo.py /path/to/dataset
```

### generate_txt.py

Generates train.txt, val.txt, and test.txt files for YOLO training.

```bash
python generate_txt.py <train_dir> <val_dir> [--test-dir <test_dir>]
```

**Arguments:**
- `train_dir`: Path to the training dataset directory
- `val_dir`: Path to the validation dataset directory
- `--test-dir`: Path to the test dataset directory (optional)

**Example:**
```bash
python generate_txt.py /path/to/train /path/to/val --test-dir /path/to/test
```

### label_format_converter.py

Converts between different label formats (e.g., XML to YOLO).

```bash
python label_format_converter.py <input_format> <output_format> <label_dir> <classes_txt>
```

**Arguments:**
- `input_format`: Input label format (e.g., xml)
- `output_format`: Output label format (e.g., txt for YOLO)
- `label_dir`: Directory containing the input labels
- `classes_txt`: Path to the classes.txt file

**Example:**
```bash
python label_format_converter.py xml txt /path/to/labels /path/to/classes.txt
```

### plot_yolo_labels.py

Visualizes YOLO format labels on images.

```bash
python plot_yolo_labels.py <dataset_dir> <classes_txt> <output_count> [options]
```

**Arguments:**
- `dataset_dir`: Path to the dataset directory
- `classes_txt`: Path to the classes.txt file
- `output_count`: Number of images to process
- `--project`: Save results to project/name (default: "runs/plot")
- `--name`: Project name (default: "exp")
- `--exist-ok`: Don't increment the run directory

**Example:**
```bash
python plot_yolo_labels.py /path/to/dataset classes.txt 10 --project results
```

### resize_img.py

Resizes images to a specified size.

```bash
python resize_img.py <input_dir> <output_dir> [--size <width> <height>]
```

**Arguments:**
- `input_dir`: Input directory containing images
- `output_dir`: Output directory for resized images
- `--size`: Target size as [width, height] (default: [640, 640])

**Example:**
```bash
python resize_img.py /path/to/input /path/to/output --size 416 416
```

### seg_augmentation.py

Applies augmentations to segmentation datasets.

```bash
python seg_augmentation.py <hyp> <dataset> [--new-image <count>]
```

**Arguments:**
- `hyp`: Path to the hyperparameters YAML file
- `dataset`: Path to the dataset directory
- `--new-image`: Number of augmented images to generate per original image

**Example:**
```bash
python seg_augmentation.py hyp.yaml /path/to/dataset --new-image 3
```

### split_val.py

Splits a dataset into training, validation, and test sets.

```bash
python split_val.py <dataset_dir> <train_percentage> <val_percentage> <test_percentage>
```

**Arguments:**
- `dataset_dir`: Path to the dataset directory
- `train_percentage`: Training set percentage (0-10)
- `val_percentage`: Validation set percentage (0-10)
- `test_percentage`: Test set percentage (0-10)

**Example:**
```bash
python split_val.py /path/to/dataset 7 2 1
```

### tweak_label.py

Modifies label classes according to predefined mappings.

```bash
python tweak_label.py <input_dir> <output_dir>
```

**Arguments:**
- `input_dir`: Input directory containing labels
- `output_dir`: Output directory for modified labels

**Example:**
```bash
python tweak_label.py /path/to/input /path/to/output
```

## Hyperparameters

Some scripts use a hyperparameters YAML file (hyp.yaml) to control augmentation settings. The default file includes:

```yaml
# Augmentation hyperparameters
fliplr: 0.5  # Horizontal flip probability
flipud: 0.0  # Vertical flip probability
scale: 0.5   # Scale factor range
translate: 0.1  # Translation factor range
rotate: 15  # Rotation angle range
shear: 10  # Shear angle range
hsv_h: 0.015  # HSV-Hue augmentation factor
hsv_s: 0.7  # HSV-Saturation augmentation factor
hsv_v: 0.4  # HSV-Value (brightness) augmentation factor
```

## Contributing

Feel free to submit issues or pull requests to improve these utilities or add new ones.
