# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

##########################
# Reusable helper functions #
##########################

# ===== Standard Library Imports =====
import sys
from pathlib import Path
# ===== Third-Party Imports =====
import cv2
import numpy as np
import torch
import ultralytics
import supervision
# ===== Own Modules =====
from settings import setting


#############################################################################################################
# MISC FUNCTIONS:

# Create all working folders in the root directory of the program if they do not exist yet.
# Returns:
#   None
def create_prg_folders() -> None:
    # https://kodify.net/python/pathlib-path-mkdir-method/
    # Folder 'models'
    Path(setting["pth_models"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_yolo_models"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_custom_models"]).mkdir(parents=True, exist_ok=True)
    # Folder 'output'
    Path(setting["pth_output"]).mkdir(parents=True, exist_ok=True)
    # Folder 'predictions'
    Path(setting["pth_predictions"]).mkdir(parents=True, exist_ok=True)
    # Folder for training images and labels
    Path(setting["pth_training_images"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_validation_images"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_training_labels"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_validation_labels"]).mkdir(parents=True, exist_ok=True)
    # Folders for image splitter
    # Input paths
    Path(setting["pth_splitin_data_img"]).mkdir(parents=True, exist_ok=True)
    Path(setting["pth_splitin_data_label"]).mkdir(parents=True, exist_ok=True)
    # Output path
    Path(setting["pth_splitout"]).mkdir(parents=True, exist_ok=True)

# Print CUDA availability and software versions.
# Returns:
#   None
def show_cuda_and_versions() -> None:
    print("\n>> DEVICE:")
    device = 'gpu (cuda)' if torch.cuda.is_available() else 'cpu (no cuda!)'
    print("Using Device:", device)
    print(">> VERSIONS:")
    print("Python: ", sys.version, "")
    print("Pytorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("Ultralytics YOLO:", ultralytics.__version__)
    print("Supervision:", supervision.__version__)
    print("Opencv:", cv2.__version__)

# Annotate an image with text using OpenCV.
# Args:
#   img (np.ndarray): Image to annotate
#   text (str): Text to draw
#   pos (tuple): (x, y) position of the text
#   font_scale (float): Font size scale
#   font_color (tuple): BGR color of the text
#   font_thickness (int): Thickness of the text
# Returns:
#   None
def annotate_text(img: np.ndarray, text: str, pos: tuple, font_scale: float,
                  font_color: tuple, font_thickness: int) -> None:
    cv2.putText(
        img,
        text,
        (pos[0], pos[1]),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=font_color,
        thickness=font_thickness,
        lineType=cv2.LINE_AA,
    )


#############################################################################################################
# USER INPUT FUNCTIONS:

# Prompt the user for a non-empty string.
# Args:
#   prompt (str): Prompt to display
# Returns:
#   str: User input
def input_empty(prompt: str) -> str:
    while(True):
        inp = input(prompt).strip()
        if(len(inp) == 0):
            print("No Input! Try again:")
        else:
            return inp

# Prompt the user for an integer.
# Args:
#   prompt (str): Prompt to display
# Returns:
#   int: User input as integer
def input_int(prompt: str) -> int:
    while(True):
        nr = input(prompt)
        if not(check_int(nr)):
            print("Input is not an integer number! Try again:")
        else:
            return int(nr)


#############################################################################################################
# VARIABLE TYPE CHECKS:

# Check if a variable can be converted to an integer.
# Args:
#   var: Variable to check
# Returns:
#   bool: True if conversion succeeds
def check_int(var) -> bool:
    try:
        val = int(var)
        return True
    except ValueError:
        return False