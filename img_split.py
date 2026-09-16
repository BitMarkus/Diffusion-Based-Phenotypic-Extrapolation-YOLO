# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

############################
# Automatic Image Splitting #
############################

# ===== Standard Library Imports =====
import os
import random
import shutil
from pathlib import Path
from random import randint
# ===== Own Modules =====
from settings import setting


class Img_Split():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self) -> None:

        # Split settings
        # Number of datasets to create
        self.num_datasets: int = setting["split_num_datasets"]
        # Extension of training images
        self.img_extension: str = setting["split_img_extension"]
        # Validation split
        self.is_val_split: bool = setting["split_is_val_split"]
        if(self.is_val_split):
            self.val_split: float = setting["split_val_split"]
        else:
            self.val_split = 0.0
        # Test split
        self.is_test_split: bool = setting["split_is_test_split"]
        if(self.is_test_split):
            self.test_split: float = setting["split_test_split"]
        else:
            self.test_split = 0.0

        # Paths
        # Input paths
        self.pth_splitin_data_img: str = setting["pth_splitin_data_img"]
        self.pth_splitin_data_label: str = setting["pth_splitin_data_label"]
        # Output path
        self.pth_splitout: str = setting["pth_splitout"]

    #############################################################################################################
    # METHODS:

    # Return a list of image names (with extension) from a folder.
    # Only includes images with the specified extension.
    # Shuffles the list randomly using the given seed.
    # Args:
    #   pth (str): Folder path
    #   extension (str): File extension to filter by
    #   seed (int): Random seed for shuffling
    # Returns:
    #   list: Shuffled list of file names, or empty list on error
    def load_img_list(self, pth: str, extension: str, seed: int) -> list:
        if(not os.path.isdir(pth)):
            print(f"Folder {pth} does not exist!")
            return []
        # Get a list of all files in the folder
        img_list = [f for f in os.listdir(pth) if os.path.isfile(os.path.join(pth, f)) and f.endswith(extension)]
        if(len(img_list) == 0):
            print(f"No images with extension {extension} in folder {pth}!")
            return []
        # Shuffle list
        random.seed(seed)
        random.shuffle(img_list)
        return img_list

    # Check that every image has a corresponding label file and warn about
    # duplicate stems (files with the same name but different extensions).
    # Args:
    #   img_list (list): List of image file names
    #   label_pth (str): Path to the label folder
    # Returns:
    #   bool: True if all images have labels and no duplicates were found
    def validate_dataset(self, img_list: list, label_pth: str) -> bool:
        # Check for duplicate stems
        stems = [os.path.splitext(f)[0] for f in img_list]
        if(len(stems) != len(set(stems))):
            print("Warning: Duplicate image stems detected! Labels may be overwritten.")
            return False
        # Check that every image has a matching label
        missing = []
        for img_file in img_list:
            stem = os.path.splitext(img_file)[0]
            label_file = f"{stem}.txt"
            if(not os.path.isfile(os.path.join(label_pth, label_file))):
                missing.append(label_file)
        if(len(missing) > 0):
            print(f"Warning: {len(missing)} images have no matching label file.")
            print(f"First few missing: {missing[:5]}")
            return False
        return True

    # Split a list into train, val, and test parts.
    # The list must already be shuffled.
    # Args:
    #   items (list): Shuffled list to split
    #   perc_val (float): Fraction of items for the validation set
    #   perc_test (float): Fraction of items for the test set
    # Returns:
    #   tuple: (train_list, val_list, test_list)
    def split_list_perc(self, items: list, perc_val: float, perc_test: float) -> tuple:
        num = len(items)
        if(num == 0):
            print("List is empty!")
            return [], [], []
        # Calculate split indices
        split_id_1 = round(num * perc_val)
        split_id_2 = round(num * perc_test) + split_id_1
        # Split list into three parts
        train_list = items[split_id_2:]
        val_list = items[:split_id_1]
        test_list = items[split_id_1:split_id_2]
        return train_list, val_list, test_list

    # Copy images from one folder to another.
    # Args:
    #   file_list (list): Names of files to copy
    #   src_pth (str): Source folder
    #   dest_pth (str): Destination folder
    # Returns:
    #   None
    def copy_img(self, file_list: list, src_pth: str, dest_pth: str) -> None:
        for img_name in file_list:
            src = os.path.join(src_pth, img_name)
            dst = os.path.join(dest_pth, img_name)
            shutil.copy(src, dst)

    # Copy label files matching the given image names.
    # Args:
    #   file_list (list): Names of images whose labels should be copied
    #   src_pth (str): Source folder for labels
    #   dest_pth (str): Destination folder for labels
    # Returns:
    #   None
    def copy_label(self, file_list: list, src_pth: str, dest_pth: str) -> None:
        for img_file in file_list:
            stem = os.path.splitext(img_file)[0]
            label_file = f"{stem}.txt"
            src = os.path.join(src_pth, label_file)
            dst = os.path.join(dest_pth, label_file)
            shutil.copy(src, dst)

    # Create the folder structure for one generated dataset.
    # Args:
    #   ds_pth (str): Path to the dataset folder
    # Returns:
    #   None
    def create_dataset_folders(self, ds_pth: str) -> None:
        # Images
        Path(os.path.join(ds_pth, "images/train/")).mkdir(parents=True, exist_ok=True)
        if(self.is_val_split):
            Path(os.path.join(ds_pth, "images/val/")).mkdir(parents=True, exist_ok=True)
        if(self.is_test_split):
            Path(os.path.join(ds_pth, "images/test/")).mkdir(parents=True, exist_ok=True)
        # Labels
        Path(os.path.join(ds_pth, "labels/train/")).mkdir(parents=True, exist_ok=True)
        if(self.is_val_split):
            Path(os.path.join(ds_pth, "labels/val/")).mkdir(parents=True, exist_ok=True)
        if(self.is_test_split):
            Path(os.path.join(ds_pth, "labels/test/")).mkdir(parents=True, exist_ok=True)

    #############################################################################################################
    # CALL:

    def __call__(self) -> None:

        # Generate random shuffle seeds (1-1000) for up to 999 datasets
        seed_list = [randint(1, 1000) for _ in range(self.num_datasets)]

        print(f"Generate {self.num_datasets} datasets. Please wait...")

        # Iterate over shuffle seeds
        for seed in seed_list:

            # Load a list of all image names in the data folder and shuffle it
            img_list = self.load_img_list(self.pth_splitin_data_img, self.img_extension, seed)

            # Skip this dataset if no images were found
            if(len(img_list) == 0):
                continue

            # Verify that labels are present for all images
            if(not self.validate_dataset(img_list, self.pth_splitin_data_label)):
                print(f"Skipping dataset {seed} due to validation errors.")
                continue

            print(f"\n> Generate dataset {seed}...")

            # Generate directory with sub directories for this dataset
            ds_pth = os.path.join(self.pth_splitout, f"dataset_{seed}/")
            # Skip if the dataset folder already exists
            if(os.path.exists(ds_pth)):
                print(f"Dataset folder {ds_pth} already exists. Skipping.")
                continue
            self.create_dataset_folders(ds_pth)

            # Split the image list
            train_list, val_list, test_list = self.split_list_perc(img_list, self.val_split, self.test_split)

            # Warn about empty splits
            if(len(train_list) == 0):
                print("Warning: Training split is empty. Skipping this dataset.")
                continue
            if(self.is_val_split and len(val_list) == 0):
                print("Warning: Validation split is empty. Consider increasing the dataset size or the val split fraction.")
            if(self.is_test_split and len(test_list) == 0):
                print("Warning: Test split is empty. Consider increasing the dataset size or the test split fraction.")

            print(f"Number of images: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

            # Copy training images and labels
            self.copy_img(train_list, self.pth_splitin_data_img, os.path.join(ds_pth, "images/train/"))
            self.copy_label(train_list, self.pth_splitin_data_label, os.path.join(ds_pth, "labels/train/"))

            # Copy validation images and labels
            if(self.is_val_split):
                self.copy_img(val_list, self.pth_splitin_data_img, os.path.join(ds_pth, "images/val/"))
                self.copy_label(val_list, self.pth_splitin_data_label, os.path.join(ds_pth, "labels/val/"))

            # Copy test images and labels
            if(self.is_test_split):
                self.copy_img(test_list, self.pth_splitin_data_img, os.path.join(ds_pth, "images/test/"))
                self.copy_label(test_list, self.pth_splitin_data_label, os.path.join(ds_pth, "labels/test/"))

            print(f"Dataset {seed} was saved in folder {ds_pth}.")

        print("\nDatasets were successfully created.")