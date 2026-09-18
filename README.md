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
8. [Dataset splitter](#dataset-splitter)
9. [Reproducing the morphological validation results](#reproducing-the-morphological-validation-results)
10. [Model checkpoints](#model-checkpoints)
11. [License](#license)
12. [Citation](#citation)

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
├── main.py                     # Entry point, console menu
├── settings.py                 # Live configuration (edit this file)
├── functions.py                # Shared helpers
├── model.py                    # YOLO model loading
├── detection.py                # Object detection logic
├── image.py                    # Image handling and result export
├── counter.py                  # Onscreen class counter
├── train.py                    # Training entry point
├── predict.py                  # Prediction entry point
├── img_split.py                # Dataset splitter
├── configs/
│   ├── nuclei.yaml             # Reference dataset config for the nuclei model
│   ├── filopodia.yaml          # Reference dataset config for the filopodia model
│   └── settings_paper.py       # Frozen settings that produced the manuscript results
├── models/                     # Model checkpoints (created at runtime, not in git)
│   ├── yolo_models/            # Pretrained YOLOv8 weights
│   └── custom_models/          # Self-trained weights
├── train/                      # Training images and labels (created at runtime)
├── predictions/                # Images for prediction (created at runtime)
├── img_splitter/               # Dataset splitter input/output (created at runtime)
└── output/                     # All program output (created at runtime)         
```

The folders `models/`, `train/`, `predictions/`, `img_splitter/`, and `output/` are created automatically on the first program start and are excluded from git.

---

## Configuration

All settings live in `settings.py` in a single dictionary called `setting`. This is the only file that needs to be edited for training or prediction.

Two settings are particularly important for switching between the two models:

- `"train_classes"` - the class dict for training. Use `{0: 'nuclei',}` for the nuclei model and `{0: 'filopodia',}` for the filopodia model.
- `"od_custom_model_name"` - the filename of the trained checkpoint to load for prediction, relative to `models/custom_models/`. For example `nuclei_DIC_best.pt` or `filopodia_DIC_best.pt`.

All other training, augmentation, and inference parameters match the values reported in the manuscript's Methods section and are documented inline in `settings.py`. The frozen version used for the manuscript is preserved in `configs/settings_paper.py` (see the end of the "Reproducing the morphological validation results" section).

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

## Dataset splitter

Creates randomized train/validation(/test) splits from a flat folder of images and their YOLO-format labels. For each requested dataset, the source images are shuffled with a distinct random seed and partitioned according to the configured ratios. The result is a set of self-contained dataset folders, each with its own `images/` and `labels/` subdirectories in the layout Ultralytics expects.

This is useful when you have a single annotated image collection and want to produce multiple independent train/val splits — for example, to verify that a reported result is not an artifact of one particular partition. This was the use case in the manuscript's pipeline: the splitter was run once on the full pool of annotated images with three datasets requested, producing three shuffled 80/20 splits (`dataset_311`, `dataset_337`, `dataset_355`). One of these (`dataset_355`) was used to train the filopodia model reported in the manuscript; the other two were used as independent splits to check that the reported performance was not specific to a single lucky partition.

### Key settings

| Setting | Type | Description | Default |
|---------|------|-------------|---------|
| `split_num_datasets` | int | Number of datasets to generate (1–999) | `3` |
| `split_img_extension` | str | File extension of the source images | `".png"` |
| `split_is_val_split` | bool | Whether to create a validation split | `True` |
| `split_val_split` | float | Fraction of images assigned to validation | `0.2` |
| `split_is_test_split` | bool | Whether to create a test split | `False` |
| `split_test_split` | float | Fraction of images assigned to test | `0.2` |

Each generated dataset uses a distinct random seed drawn from `[1, 999]`. The seed is used both as the shuffle seed and as part of the output folder name, so it is always visible which seed produced which dataset.

If both validation and test splits are enabled, the split is applied in the order train → val → test, with train receiving the remainder. For example, with 100 images, `split_val_split = 0.2` and `split_test_split = 0.1`, the resulting split is 70 train, 20 val, 10 test.

### Input folder structure

```plaintext
img_splitter/
├── input/
│   ├── images/                  # Source images (all in one flat folder)
│   │   ├── img_0001.png
│   │   ├── img_0002.png
│   │   └── ...
│   └── labels/                  # YOLO-format labels matching the images
│       ├── img_0001.txt
│       ├── img_0002.txt
│       └── ...
```

Every image in `input/images/` must have a matching label file in `input/labels/`, sharing the same stem (e.g. `img_0001.png` pairs with `img_0001.txt`). The splitter checks this before writing any files and reports any missing or duplicate stems.

The folder structure is created automatically on the first program start if it does not exist. Images and labels must be placed there manually before running the splitter.

### Output folder structure

Each dataset is written to its own self-contained folder under `img_splitter/output/`:

```plaintext
img_splitter/output/
├── dataset_311/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── dataset_337/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── dataset_355/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

When `split_is_test_split = True`, each dataset also gets an `images/test/` and `labels/test/` folder.

Each dataset folder is fully self-contained and uses the same layout Ultralytics expects.

### Running the splitter

1. Place the source images in `img_splitter/input/images/` and the matching YOLO label files in `img_splitter/input/labels/`.

2. Set the splitter settings in `settings.py` (see the table above).

3. Run the program:

   ```bash
   python main.py
   ```

4. Select option `3) Dataset splitter` from the menu.

Typical console output:

```plaintext
:DATASET SPLITTER:
  Input: img_splitter/input/
  Output: img_splitter/output/
  Number of datasets to generate: 3
  Validation split: 20%

Generate 3 datasets. Please wait...

> Generate dataset 311...
Number of images: train=80, val=20, test=0
Dataset 311 was saved in folder img_splitter/output/dataset_311/.

> Generate dataset 337...
Number of images: train=80, val=20, test=0
Dataset 337 was saved in folder img_splitter/output/dataset_337/.

> Generate dataset 355...
Number of images: train=80, val=20, test=0
Dataset 355 was saved in folder img_splitter/output/dataset_355/.

Datasets were successfully created.
```

### Using a generated dataset for training

Point the training paths in `settings.py` at the corresponding folder:

```python
pth_training_images = "img_splitter/output/dataset_355/images/train/"
pth_validation_images = "img_splitter/output/dataset_355/images/val/"
pth_training_labels = "img_splitter/output/dataset_355/labels/train/"
pth_validation_labels = "img_splitter/output/dataset_355/labels/val/"
```

Then run training as described in the Training section.

Alternatively, copy the contents of a generated dataset into `train/`, which the training code reads by default.

### Notes

- The splitter operates on a **flat folder of images**. If your source images are in nested subfolders, flatten them first with a simple move-and-rename step.
- The splitter does not modify the source folder; images are copied, not moved.
- Filenames are preserved exactly, so any previous renaming (e.g. adding confidence suffixes) is unaffected.
- The splitter verifies that every image has a matching label and that no two images share a stem before writing any files. If a check fails, it skips the affected dataset and continues with the remaining ones.
- If a dataset folder with the same seed already exists, the splitter skips that dataset rather than overwriting it. To regenerate a dataset, delete its folder from `img_splitter/output/` first.
- The same source image may appear in different datasets. The splitter draws from a single source pool, so different random seeds produce overlapping (but not identical) train/val partitions. This is intentional for the "multiple independent splits" use case — the datasets share underlying data but partition it differently.

---

## Reproducing the morphological validation results

The quantitative validation of morphological features in the manuscript was produced with the following settings, which are the defaults in `settings.py`:

| Setting                  | Value   |
| ------------------------ | ------- |
| Model architecture       | YOLOv8m |
| Training image size      | 512     |
| Training epochs          | 1000    |
| Batch size               | 32      |
| Optimizer                | auto (Ultralytics resolves to AdamW) |
| Initial learning rate    | 0.002 (effective, selected by auto)  |
| Final learning rate frac | 0.01    |
| Momentum                 | 0.9 (effective, selected by auto)    |
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

> **Note on the optimizer setting.** In `settings.py`, the training optimizer is
> set to `"train_optimizer": "auto"`. With this setting, Ultralytics selects the
> optimizer and its hyperparameters automatically. For the dataset and model used
> in the manuscript, it selects AdamW with an initial learning rate of 0.002 and a
> momentum of 0.9. These are the values reported in the manuscript's Methods
> section, even though `settings.py` lists different values under `train_lr0`
> and `train_momentum` (those are ignored when `auto` is active and are retained
> only as documentation).

For the manuscript, the `best.pt` checkpoint from each training run was used. Because Ultralytics does not record the exact epoch of the best checkpoint in the weights file, the epoch is not reported.

To reproduce the results:

1. Train the nuclei model with `"train_classes": {0: 'nuclei',}`. Save the resulting `best.pt` as `models/custom_models/nuclei_DIC_best.pt`.
2. Train the filopodia model with `"train_classes": {0: 'filopodia',}`. Save the resulting `best.pt` as `models/custom_models/filopodia_DIC_best.pt`.
3. Set `"od_custom_model_name"` to the desired model, place the corresponding images in `predictions/`, and run option `2) Predict on images`.
4. Use the resulting per-image bounding box text files to compute counts of nuclei or filopodia per image.

The absolute counts in `results.txt` are the input to the frame-wise counts and the statistical comparisons between real WT and KO images presented in the manuscript.

### Frozen paper settings

The exact configuration that produced the manuscript's results is preserved in
`configs/settings_paper.py`. The live `settings.py` file is the one the program
actually reads, and it may drift from the frozen version over time — for example,
while testing new features or adjusting paths for a local setup.

If you want to verify that your configuration matches the paper exactly, diff
the two files:

```bash
diff settings.py configs/settings_paper.py
```

Or, to restore the paper configuration, copy the frozen file to the repository
root and rename it:

```bash
cp configs/settings_paper.py settings.py
```

The frozen file is a complete, drop-in configuration: every setting the current code reads is defined, including the dataset splitter settings that were used to generate the paper's training splits. Values that differ from the exact paper run — for example, utilities that did not exist at the time — are flagged with a
comment starting with `# PAPER:` in the frozen file.

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