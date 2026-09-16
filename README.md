# Diffusion-Based-Phenotypic-Extrapolation-YOLO

YOLOv8-based detection of nuclei and filopodia in DIC microscopy images of human fibroblasts. This repository is the companion code for the YOLO-based morphological validation presented in the *Diffusion-Based Phenotypic Extrapolation* manuscript.

It is a trimmed, paper-focused subset of a larger general-purpose YOLO framework (training, detection, segmentation, tracking, and screen capture). Only the pieces required to reproduce the nuclei and filopodia detection results in the manuscript are included here.

**This is not a general-purpose software package.** It documents the code used to produce the manuscript's morphological validation results and the corresponding quantitative statements in the Results section, and enables reproduction of those results.

---

## Index

1. [Overview](#overview)
2. [Installation](#installation)
3. [Repository structure](#repository-structure)
4. [Configuration](#configuration)
5. [Key settings](#key-settings)
6. [Training](#training)
7. [Prediction](#prediction)
8. [Reproducing the morphological validation results](#reproducing-the-morphological-validation-results)
9. [Model checkpoints](#model-checkpoints)
10. [License](#license)
11. [Citation](#citation)

---

## Overview

The manuscript uses two YOLOv8m object detection models:

- A **nuclei** detection model, trained on 900 manually annotated DIC images containing approximately 1,700 nuclei.
- A **filopodia** detection model, trained on 100 manually annotated DIC images containing approximately 6,000 filopodia. Annotations focused specifically on small, hair-like filopodia rather than larger, longer protrusions.

Both models were used to quantify morphological differences between wild-type (WT) and knockout (KO) fibroblasts in real and synthetic DIC images, producing the quantitative validation of morphological features presented in the manuscript.

---

## Installation

This repository is designed to slot into an existing PyTorch environment. It has been tested with Python 3.8 and PyTorch 2.4.1 (CUDA 12.1), matching the environment used for the manuscript.

Install PyTorch with the appropriate CUDA variant for your system first. For CUDA 12.1:

```bash
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## Repository structure

```
Diffusion-Based-Phenotypic-Extrapolation-YOLO/
├── main.py                 # Entry point, console menu
├── settings.py             # All configuration
├── functions.py            # Shared helpers
├── model.py                # YOLO model loading
├── detection.py            # Object detection logic
├── image.py                # Image handling and result export
├── counter.py              # Onscreen class counter
├── train.py                # Training entry point
├── predict.py              # Prediction entry point
├── configs/
│   ├── nuclei.yaml         # Reference dataset config for the nuclei model
│   └── filopodia.yaml      # Reference dataset config for the filopodia model
├── models/                 # Model checkpoints (created at runtime, not in git)
│   ├── yolo_models/        # Pretrained YOLOv8 weights
│   └── custom_models/      # Self-trained weights
├── train/                  # Training images and labels (created at runtime)
├── predictions/            # Images for prediction (created at runtime)
└── output/                 # All program output (created at runtime)
```

The folders `models/`, `train/`, `predictions/`, and `output/` are created automatically on the first program start and are excluded from git.

---

## Configuration

All settings live in `settings.py` in a single dictionary called `setting`. This is the only file that needs to be edited for training or prediction.

Two settings are particularly important for switching between the two models:

- `"train_classes"` - the class dict for training. Use `{0: 'nuclei',}` for the nuclei model and `{0: 'filopodia',}` for the filopodia model.
- `"od_custom_model_name"` - the filename of the trained checkpoint to load for prediction, relative to `models/custom_models/`. For example `nuclei_DIC_best.pt` or `filopodia_DIC_best.pt`.

All other training, augmentation, and inference parameters match the values reported in the manuscript's Methods section and are documented inline in `settings.py`.

---

## Key settings

The settings below are the ones a reader is most likely to need. They are grouped by purpose. Every setting in `settings.py` has an inline comment describing its effect.

### Model selection

| Setting                    | Description                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `od_use_pretrained_model`  | Set to `True` to use a stock YOLOv8 checkpoint, `False` to load a self-trained checkpoint from `models/custom_models/`.            |
| `od_custom_model_name`     | Filename of the self-trained checkpoint to load when `od_use_pretrained_model` is `False`. Selects nuclei vs. filopodia model.     |
| `od_pretrained_model_size` | Size of the pretrained model to load when `od_use_pretrained_model` is `True` (`n`, `s`, `m`, `l`, `x`).                           |

### Training

| Setting                    | Description                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `train_classes`            | Class dict for training. Set to `{0: 'nuclei',}` for the nuclei model or `{0: 'filopodia',}` for the filopodia model.             |
| `train_model_size`         | YOLOv8 size used for training (`n`, `s`, `m`, `l`, `x`). Both manuscript models used `m`.                                          |
| `train_num_epochs`         | Total number of training epochs.                                                                                                  |
| `train_batch_size`         | Batch size. Lower this if you hit out-of-memory errors on a smaller GPU.                                                          |
| `train_img_size`           | Image size used during training. Both manuscript models used `512`.                                                                |
| `train_seed`               | Random seed for training. Both manuscript models used `111`.                                                                      |
| `train_chckpt_save_period` | Interval (in epochs) at which intermediate checkpoints are saved. Set to `-1` to disable.                                          |

### Inference

| Setting                    | Description                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `od_inf_size_img`          | Image size used during inference. Both manuscript models used `(512, 512)`.                                                       |
| `od_min_conf`              | Minimum confidence threshold for detections. Manuscript used `0.4`.                                                               |
| `od_iou`                   | IoU threshold for non-maximum suppression. Manuscript used `0.8`.                                                                 |
| `od_agnostic_nms`          | If `True`, suppresses overlapping boxes across classes. Manuscript used `True`.                                                    |
| `od_max_detections`        | Maximum number of detections per image. Manuscript used `500`.                                                                    |
| `od_class_selection`       | List of class IDs to keep. Empty list means all classes are kept.                                                                  |

### Output

| Setting                    | Description                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `od_export_results`        | If `True`, writes a summary `results.txt` with per-image class counts to `output/`.                                                |
| `od_save_bbox_txt`         | If `True`, writes a YOLO-format `.txt` file with bounding boxes per image to `output/`.                                            |
| `od_save_predicted_images` | If `True`, saves annotated images to `output/`.                                                                                    |
| `od_show_predicted_images` | If `True`, opens a window showing each annotated image. Press any key to advance.                                                  |
| `od_show_class_counter`    | If `True`, draws an on-screen count of detections per class on each image.                                                         |

### Display

| Setting                    | Description                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `od_show_bbox`             | If `True`, draws bounding boxes on the output images.                                                                              |
| `od_show_labels`           | If `True`, draws the class label and confidence above each bounding box.                                                            |
| `od_bbox_line_thickness`   | Line thickness of bounding boxes.                                                                                                  |
| `od_bbox_text_scale`       | Font size for labels and confidences.                                                                                              |

### Paths

| Setting                    | Description                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `pth_custom_models`        | Folder where self-trained checkpoints are placed. Default is `models/custom_models/`.                                              |
| `pth_predictions`          | Folder containing images to run prediction on. Default is `predictions/`.                                                          |
| `pth_output`               | Folder where all output (result files, saved images, training runs) is written. Default is `output/`.                              |

---

## Training

1. Prepare the training data in `train/images/train/` and the labels in `train/labels/train/`. Validation images and labels go into `train/images/val/` and `train/labels/val/`. Labels must be in YOLO format (`class_id x_center y_center width height`, normalized 0-1).

2. Set `"train_classes"` in `settings.py` for the model you want to train.

3. Run the program:

   ```bash
   python main.py
   ```

4. Select option `1) Train model`.

The dataset configuration (`train/config.yaml`) is generated automatically from `settings.py` and does not need to be edited by hand. The two YAML files in `configs/` are provided as a reference for what that generated file looks like for each model.

Training checkpoints are saved to `output/train/` (or `output/train2/`, etc. for subsequent runs, following Ultralytics' default folder naming). The best checkpoint is named `best.pt` and the most recent one `last.pt`. Additional checkpoints are saved every 50 epochs, as controlled by `"train_chckpt_save_period"`.

---

## Dataset splitter

If you want to retrain on your own annotated data, the repository includes a
utility that creates randomized train/val/test splits from a flat folder of
images and labels.

1. Place all images in `img_splitter/input/images/` and their matching YOLO-format
   labels in `img_splitter/input/labels/`. File stems must match (e.g.
   `img_001.png` pairs with `img_001.txt`).

2. Set the following in `settings.py`:

   - `"split_num_datasets"` - how many datasets to generate (each uses a different
     random seed).
   - `"split_is_val_split"` and `"split_val_split"` - whether to create a validation
     split, and what fraction of images go into it.
   - `"split_is_test_split"` and `"split_test_split"` - same for a test split.

3. Run the program (`python main.py`) and select option `3) Dataset splitter`.

Each generated dataset is written to `img_splitter/output/dataset_<seed>/` and is
self-contained:

```
img_splitter/output/dataset_<seed>/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

To train on a generated dataset, either copy its contents into `train/`, or point
`"pth_training_images"` and `"pth_validation_images"` in `settings.py` at the
corresponding folders inside `img_splitter/output/dataset_<seed>/`.

The splitter validates that every image has a matching label and that no two images
share the same stem before writing any files, so a corrupted input folder is caught
early rather than after a partial copy.

---

## Prediction

1. Place the images you want to process in `predictions/`.

2. Put the trained checkpoint into `models/custom_models/` and set `"od_custom_model_name"` in `settings.py` to match.

3. Run the program:

   ```bash
   python main.py
   ```

4. Select option `2) Predict on images`.

Two output files are written to `output/`:

- `results.txt` - a summary table with one row per image and one column per class, plus the inference parameters (checkpoint, minimum confidence, IoU threshold) at the top.
- One `.txt` file per image, in YOLO format, containing the bounding boxes for that image. This can be disabled by setting `"od_save_bbox_txt"` to `False`.

Predicted images themselves are not saved by default. To save annotated images, set `"od_save_predicted_images"` to `True`. To view them on screen before saving, also set `"od_show_predicted_images"` to `True`.

---

## Reproducing the morphological validation results

The quantitative validation of morphological features in the manuscript was produced with the following settings, which are the defaults in `settings.py`:

| Setting                  | Value   |
| ------------------------ | ------- |
| Model architecture       | YOLOv8m |
| Training image size      | 512     |
| Training epochs          | 1000    |
| Batch size               | 32      |
| Optimizer                | AdamW   |
| Initial learning rate    | 0.005   |
| Final learning rate frac | 0.01    |
| Momentum                 | 0.937   |
| Weight decay             | 0.0005  |
| Warmup epochs            | 5.0     |
| Box loss weight          | 7.5     |
| Class loss weight        | 0.5     |
| Shuffle seed             | 111     |
| Inference image size     | 512x512 |
| Minimum confidence       | 0.4     |
| IoU threshold            | 0.8     |
| Agnostic NMS             | True    |
| Maximum detections       | 500     |

The two models were trained independently with the same seed. Both used the same architecture and hyperparameters; only the training dataset and the class name (`nuclei` vs. `filopodia`) differed.

For the manuscript, the `best.pt` checkpoint from each training run was used. Because Ultralytics does not record the exact epoch of the best checkpoint in the weights file, the epoch is not reported.

To reproduce the results:

1. Train the nuclei model with `"train_classes": {0: 'nuclei',}`. Save the resulting `best.pt` as `models/custom_models/nuclei_DIC_best.pt`.
2. Train the filopodia model with `"train_classes": {0: 'filopodia',}`. Save the resulting `best.pt` as `models/custom_models/filopodia_DIC_best.pt`.
3. Set `"od_custom_model_name"` to the desired model, place the corresponding images in `predictions/`, and run option `2) Predict on images`.
4. Use the resulting per-image bounding box text files to compute counts of nuclei or filopodia per image.

The absolute counts in `results.txt` are the input to the frame-wise counts and the statistical comparisons between real WT and KO images presented in the manuscript.

---

## Model checkpoints

The trained model checkpoints are **not** distributed with this repository. They are available from the corresponding author upon reasonable request:

**Markus Reichold** - markus.reichold@ur.de

Requests should specify whether the nuclei model, the filopodia model, or both are needed, and the intended use.

---

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

Note: this repository depends on Ultralytics YOLOv8, which is itself distributed under AGPL-3.0. The AGPL-3.0 license of this repository is consistent with that dependency.

---

## Citation

If you use this code, please cite the manuscript:

> [Author list]. *Diffusion-Based Phenotypic Extrapolation*. [Journal], [Year]. DOI: [to be added]

A full citation and DOI will be added once the manuscript is published.

---

## Related repositories

- **Diffusion-Based-Phenotypic-Extrapolation** - the main companion repository containing the CNN classification pipeline, diffusion-based morphing series generation, and analysis code: https://github.com/BitMarkus/Diffusion-Based-Phenotypic-Extrapolation

- **YOLO-Object-Detection** - the general-purpose YOLO framework from which the code in this repository was derived. It additionally supports object segmentation, tracking, video inference, and screen capture: https://github.com/BitMarkus/YOLOv8-Object-Detection-and-Segmentation