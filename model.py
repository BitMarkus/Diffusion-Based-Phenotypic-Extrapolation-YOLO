# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

######################
# Handling of models #
######################

# ===== Standard Library Imports =====
import os
# ===== Third-Party Imports =====
from ultralytics import YOLO
# ===== Own Modules =====
from settings import setting

class ModelOD():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, pth: str | None = None) -> None:

        # If no path to a specific model is given in the constructor,
        # the path/model is determined by the settings
        if(pth is None):
            # Select model for detection (pretrained vs. custom)
            self.use_pretrained_model: bool = setting["od_use_pretrained_model"]
            # Model name strings
            self.model_prefix: str = "yolov8"
            self.pretrained_model_extension: str = ".pt"
            # Load model
            # Pretrained model
            if(self.use_pretrained_model):
                self.model_name: str = f'{self.model_prefix}{setting["od_pretrained_model_size"]}{self.pretrained_model_extension}'
                # Model with path
                self.model_pth: str = f'{setting["pth_yolo_models"]}{self.model_name}'
            # Custom model
            else:
                self.model_name = setting["od_custom_model_name"]
                # Model with path
                self.model_pth = f'{setting["pth_custom_models"]}{self.model_name}'

        # If a path to a model is specified in the constructor,
        # the settings will be ignored and the model in the specified path will be loaded
        else:
            # Get model name from path
            # https://stackoverflow.com/questions/3925096/how-to-get-only-the-last-part-of-a-path-in-python
            self.model_pth = pth
            self.model_name = os.path.basename(os.path.normpath(self.model_pth))

        self.model: YOLO = self.load_model(self.model_pth)
        # Get a dict with all class names the model was trained on
        self.class_names: dict = self.model.model.names

    #############################################################################################################
    # METHODS:

    # Load model: Pretrained or custom trained.
    # Args:
    #   model_name (str): Path to the model file
    # Returns:
    #   YOLO: Loaded YOLO model
    def load_model(self, model_name: str) -> YOLO:
        # Load YOLOv8 model
        model = YOLO(model_name)
        # fuse() optimizes the model by fusing Conv2d and BatchNorm2d layers,
        # which can improve inference speed
        model.fuse()
        return model

    # Method to print the classes of a model.
    # Returns:
    #   None
    def print_classes(self) -> None:
        print("Model name:", self.model_pth)
        print("Number of classes:", len(self.class_names))
        for key, value in self.class_names.items():
            print(f"[{key}]: {value}")