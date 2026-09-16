# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

#####################
# Network training  #
#####################

# ===== Standard Library Imports =====
import os
import shutil
# ===== Third-Party Imports =====
from ultralytics import YOLO
# ===== Own Modules =====
from settings import setting


class Train():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self) -> None:
        
        # Path to dataset info
        self.pth_dataset_info: str = f'{setting["pth_dataset_info"]}{setting["train_dataset_info_name"]}'
        # Path to training output
        self.pth_training_output: str = setting["pth_output"]
        # Selection of model for training
        # Either an empty model or a pretrained model (yolov8 models) can be used for training
        # Model name strings
        self.model_prefix: str = "yolov8"
        self.empty_model_extension: str = ".yaml"
        self.pretrained_model_extension: str = ".pt"
        # Set model for training
        self.use_pretrained_model: bool = setting["train_use_pretrained_model"]
        self.model_name: str = f'{self.model_prefix}{setting["train_model_size"]}'
        if(self.use_pretrained_model):
            self.model_file = f'{self.model_name}{self.pretrained_model_extension}'
        else:
            self.model_file = f'{self.model_name}{self.empty_model_extension}'
        self.model = YOLO(f'{setting["pth_yolo_models"]}{self.model_file}')
        # Network hyperparameters
        self.num_epochs: int = setting["train_num_epochs"]
        self.batch_size: int = setting["train_batch_size"]
        self.patience: int = setting["train_patience"]
        # Image size and shape for training
        self.train_img_size: int = setting["train_img_size"]
        self.train_rectangular_img: bool = setting["train_rectangular_img"]
        # Train validation
        self.do_validation: bool = setting["train_validation"]
        self.verbose: bool = setting["train_verbose"]
        self.seed: int = setting["train_seed"]
        self.save_plots: bool = setting["train_save_plots"]
        # Save checkpoints each x epochs
        self.chckpt_save_period: int = setting["train_chckpt_save_period"]
        # Emphasize box or class accuracy
        self.box_emph: float = setting["train_box_emph"]
        self.class_emph: float = setting["train_class_emph"]
        # Dataset parameters
        self.train_classes: dict = setting["train_classes"]
        self.pth_training_images: str = setting["pth_training_images"]
        self.pth_validation_images: str = setting["pth_validation_images"]
        # Augmentation parameters
        self.aug_hsv_h: float = setting["aug_hsv_h"]
        self.aug_hsv_s: float = setting["aug_hsv_s"]
        self.aug_hsv_v: float = setting["aug_hsv_v"]
        self.aug_degrees: float = setting["aug_degrees"]
        self.aug_translate: float = setting["aug_translate"]
        self.aug_scale: float = setting["aug_scale"]
        self.aug_shear: float = setting["aug_shear"]
        self.aug_perspective: float = setting["aug_perspective"]
        self.aug_flipud: float = setting["aug_flipud"]
        self.aug_fliplr: float = setting["aug_fliplr"]
        self.aug_mosaic: float = setting["aug_mosaic"]
        self.aug_mixup: float = setting["aug_mixup"]
        # Training hyperparameters
        self.hyp_optimizer: str = setting["train_optimizer"]
        self.hyp_cos_lr: bool = setting["train_cos_lr"]
        self.hyp_lr0: float = setting["train_lr0"]
        self.hyp_lrf: float = setting["train_lrf"]
        self.hyp_momentum: float = setting["train_momentum"]
        self.hyp_weight_decay: float = setting["train_weight_decay"]
        self.hyp_warmup_epochs: float = setting["train_warmup_epochs"]
        self.hyp_warmup_momentum: float = setting["train_warmup_momentum"]
        self.hyp_warmup_bias_lr: float = setting["train_warmup_bias_lr"]
        self.hyp_dropout: float = setting["train_dropout"]

    #############################################################################################################
    # METHODS:

    # Dynamically create the dataset.yaml file, which is necessary for training.
    # Returns:
    #   None
    def generate_dataset_info(self) -> None:
        # Get the program's root directory path
        home = os.getcwd()
        # Create a new dataset.yaml file
        # This mode opens the file for writing only (overwrites existing content)
        with open(self.pth_dataset_info, "w") as f:
            # Write home and image directories
            f.write(f'path: {home}\n')
            f.write(f'train: {self.pth_training_images}\n')
            f.write(f'val: {self.pth_validation_images}\n\n')
            # Write classes
            f.write('names:\n')
            for key in self.train_classes:
                f.write(f'  {key}: {self.train_classes[key]}\n')

    # Delete the dataset.yaml file.
    # Returns:
    #   None
    def delete_dataset_info(self) -> None:
        if os.path.exists(self.pth_dataset_info):
            os.remove(self.pth_dataset_info)

    # Copy the settings file and the generated dataset config into the
    # training output folder for documentation. Ultralytics names its
    # output folder automatically (train, train2, ...), so this is done
    # after training has finished, once the folder name is known.
    # Args:
    #   save_dir (str): Ultralytics training output folder
    # Returns:
    #   None
    def copy_train_metadata(self, save_dir: str) -> None:
        # Copy settings.py with a "_copy" suffix to avoid accidental imports
        settings_src = "settings.py"
        settings_dst = os.path.join(save_dir, "settings_copy.py")
        if(os.path.exists(settings_src)):
            shutil.copy(settings_src, settings_dst)
            print(f"Copied settings.py to {settings_dst}")

        # Copy the dataset config generated by this training run
        config_src = self.pth_dataset_info
        config_dst = os.path.join(save_dir, "config_copy.yaml")
        if(os.path.exists(config_src)):
            shutil.copy(config_src, config_dst)
            print(f"Copied {config_src} to {config_dst}")

    #############################################################################################################
    # CALL:

    def __call__(self) -> None:

        # Generate dataset.yaml file
        self.generate_dataset_info()

        # Print the key training settings so they are visible in the log
        print(f"Training with optimizer={self.hyp_optimizer}, lr0={self.hyp_lr0}, "
              f"momentum={self.hyp_momentum}, weight_decay={self.hyp_weight_decay}")

        # Train the model
        # Train parameters: https://docs.ultralytics.com/usage/cfg/#train-settings
        self.model.train(
            model=self.model,
            data=self.pth_dataset_info,
            pretrained=self.use_pretrained_model,
            imgsz=self.train_img_size,
            patience=self.patience,
            val=self.do_validation,
            verbose=self.verbose,
            seed=self.seed,
            plots=self.save_plots,
            project=self.pth_training_output,
            rect=self.train_rectangular_img,
            save_period=self.chckpt_save_period,
            box=self.box_emph,
            cls=self.class_emph,
            save=True,
            # Training hyperparameters
            epochs=self.num_epochs,
            batch=self.batch_size,
            optimizer=self.hyp_optimizer,
            cos_lr=self.hyp_cos_lr,
            lr0=self.hyp_lr0,
            lrf=self.hyp_lrf,
            momentum=self.hyp_momentum,
            weight_decay=self.hyp_weight_decay,
            warmup_epochs=self.hyp_warmup_epochs,
            warmup_momentum=self.hyp_warmup_momentum,
            warmup_bias_lr=self.hyp_warmup_bias_lr,
            dropout=self.hyp_dropout,
            # Augmentation parameters
            hsv_h=self.aug_hsv_h,
            hsv_s=self.aug_hsv_s,
            hsv_v=self.aug_hsv_v,
            degrees=self.aug_degrees,
            translate=self.aug_translate,
            scale=self.aug_scale,
            shear=self.aug_shear,
            perspective=self.aug_perspective,
            flipud=self.aug_flipud,
            fliplr=self.aug_fliplr,
            mosaic=self.aug_mosaic,
            mixup=self.aug_mixup,
        )

        # Copy settings and config to the training output folder
        # for documentation purposes
        self.copy_train_metadata(self.model.trainer.save_dir)