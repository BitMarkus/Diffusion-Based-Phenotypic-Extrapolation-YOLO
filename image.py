# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

######################
# Handling of images #
######################

# ===== Standard Library Imports =====
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
# ===== Third-Party Imports =====
import cv2
import numpy as np
from PIL import Image
# ===== Own Modules =====
from detection import Detect
from model import ModelOD
from settings import setting


class ImageOD():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self) -> None:

        # Path for images to predict
        self.pth_predictions = setting["pth_predictions"]
        # Output path for saved images and result files
        self.pth_image_output = setting["pth_output"]
        # Show predicted images
        self.show_pred_images = setting["od_show_predicted_images"]
        # Save predicted images
        self.save_pred_images = setting["od_save_predicted_images"]
        # Result window: position and title
        self.window_results_title = setting["window_results_title"]
        self.window_results_x_pos = setting["window_results_x_pos"]
        self.window_results_y_pos = setting["window_results_y_pos"]
        # Result export
        self.export_results = setting["od_export_results"]
        self.export_file_name = setting["od_export_file_name"]
        # Per-image bounding box txt export
        self.save_bb_txt = setting["od_save_bbox_txt"]
        # Parameters for predictions (for result export)
        self.iou = setting["od_iou"]
        self.min_conf = setting["od_min_conf"]
        # Model path (set during __call__)
        self.model_pth: Optional[str] = None

    #############################################################################################################
    # METHODS:

    # Return a list of all image files in a folder.
    # Args:
    #   pth (str): Folder path
    # Returns:
    #   List[str]: List of file names
    def get_img_list(self, pth: str) -> List[str]:
        images = [img for img in os.listdir(pth) if os.path.isfile(os.path.join(pth, img))]
        return images

    # Load an image from disk.
    # Args:
    #   name (str): Image file name
    #   pth (str): Folder path
    # Returns:
    #   np.ndarray: Loaded image
    def load_image(self, name: str, pth: str) -> np.ndarray:
        img_pth = f'{pth}{name}'
        img = cv2.imread(img_pth)
        return img

    # Save a predicted image to disk.
    # Args:
    #   img (np.ndarray): Image to save
    #   name (str): Output file name
    #   output_pth (str): Output folder
    # Returns:
    #   None
    def save_image(self, img: np.ndarray, name: str, output_pth: str) -> None:
        # Change color to RGB
        img = Image.fromarray(img[:, :, ::-1])
        img_pth = f'{output_pth}{name}'
        img.save(img_pth)
        print(f'Image {name} was saved to folder {output_pth}.')

    # Display a predicted image. Closes on any keypress.
    # Args:
    #   img (np.ndarray): Image to display
    # Returns:
    #   None
    def show_image(self, img: np.ndarray) -> None:
        cv2.namedWindow(self.window_results_title)
        cv2.moveWindow(self.window_results_title, self.window_results_x_pos, self.window_results_y_pos)
        cv2.imshow(self.window_results_title, img)
        # Close image window by pressing any key
        cv2.waitKey(0)
        # Close all open windows
        cv2.destroyAllWindows()

    # Write the summary detection results to a text file.
    # Args:
    #   results (Dict[str, Dict[str, int]]): Per-image class counts
    #   output_pth (str): Output folder
    # Returns:
    #   None
    def save_result_file(self, results: Dict[str, Dict[str, int]], output_pth: str) -> None:
        file_pth = f"{output_pth}{self.export_file_name}"
        # Open the file for writing only (overwrites existing content)
        with open(file_pth, "w") as f:
            # Get first index in result dict
            first_idx = next(iter(results))
            # Get number of classes
            num_classes = len(results[first_idx])

            # Write hyperparameters for prediction
            f.write("----------- PREDICTION PARAMETERS -----------\n")
            f.write("\n")
            f.write(f"Checkpoint: {self.model_pth}\n")
            f.write(f"Min conf: {self.min_conf}\n")
            f.write(f"IoU: {self.iou}\n")
            f.write("\n")

            # Write detections table
            f.write("----------- DETECTIONS -----------\n")
            f.write("\n")
            # Write header
            f.write("Image name, ")
            counter = 0
            for class_name in results[first_idx]:
                f.write(f"Num {class_name}")
                counter += 1
                if(counter < num_classes):
                    f.write(", ")
                else:
                    f.write(":\n")

            # Write one row per image
            for img in results:
                f.write(f"{img}, ")
                counter = 0
                for class_name in results[img]:
                    f.write(f"{results[img][class_name]}")
                    counter += 1
                    if(counter < num_classes):
                        f.write(", ")
                    else:
                        f.write("\n")

    #############################################################################################################
    # CALL:

    def __call__(self) -> None:

        # Load model and class names for prediction
        model_object = ModelOD()
        self.model_pth = model_object.model_pth

        # Object detection
        self.detection = Detect(model_object)

        # Load images in prediction folder
        images = self.get_img_list(self.pth_predictions)

        # If there are no images in the folder
        if(len(images) == 0):
            print(f'No images were found in folder {self.pth_predictions}!')
            return

        print(f"There are {len(images)} images for prediction in the prediction folder.")

        # Create a dict to store class counts
        if(self.export_results):
            results: Dict[str, Dict[str, int]] = {}

        # Iterate over images
        for image_name in images:
            # Load image
            img = self.load_image(image_name, self.pth_predictions)

            # Object detection
            img, result, raw_result = self.detection(img)

            # Store per-image class counts
            if(self.export_results):
                results[image_name] = result

            # Write per-image bounding box txt
            if(self.save_bb_txt):
                image_stem = os.path.splitext(image_name)[0]
                bb_pth = f"{self.pth_image_output}{image_stem}.txt"
                self.detection.write_bbox_txt(raw_result, bb_pth)

            # Show image
            if(self.show_pred_images):
                self.show_image(img)

            # Save image
            if(self.save_pred_images):
                self.save_image(img, image_name, self.pth_image_output)

        # Save summary results in a text file
        if(self.export_results):
            self.save_result_file(results, self.pth_image_output)