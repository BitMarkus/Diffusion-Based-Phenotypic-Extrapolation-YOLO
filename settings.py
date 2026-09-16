# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

####################
# Program settings #
####################

setting = {

    ############
    # TRAINING #
    ############

    # Determines whether to start training from a pretrained model.
    # Can be a boolean value or a string path to a specific model from which to load weights
    "train_use_pretrained_model": True,
    # Size of the pretrained model (n, s, m, l, x)
    "train_model_size": "m",
    # Name of the necessary train info located in folder /train
    "train_dataset_info_name": "config.yaml",
    # Dict of classes for automatic generation of the dataset.yaml file
    # For the nuclei model: {0: 'nuclei',}
    # For the filopodia model: {0: 'filopodia',}
    "train_classes": {0: 'filopodia',},

    # Training parameters
    # https://docs.ultralytics.com/usage/cfg/#train-settings
    # Total number of training epochs. Each epoch represents a full pass over the entire dataset
    "train_num_epochs": 1000,
    # Batch size, with three modes: set as an integer (e.g., batch=16), auto mode for 60% GPU memory
    # utilization (batch=-1), or auto mode with specified utilization fraction (batch=0.70)
    "train_batch_size": 32,
    # Number of epochs to wait without improvement in validation metrics before early stopping the training.
    # 0 for no early stopping
    "train_patience": 0,
    # Enables validation during training, allowing for periodic evaluation of model performance on a separate dataset
    "train_validation": True,
    # Target image size for training. All images are resized to this dimension before being fed into the model
    "train_img_size": 512,
    # Enables rectangular training, optimizing batch composition for minimal padding.
    # TRAINING WARNING: 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False
    "train_rectangular_img": False,
    # Enables verbose output during training, providing detailed logs and progress updates
    "train_verbose": True,
    # Sets the random seed for training, ensuring reproducibility of results across runs with the same configurations
    "train_seed": 111,
    # Generates and saves plots of training and validation metrics, as well as prediction examples,
    # providing visual insights into model performance and learning progression
    "train_save_plots": True,
    # Frequency of saving model checkpoints, specified in epochs. A value of -1 disables this feature
    "train_chckpt_save_period": 50,
    # Weight of the box loss component in the loss function, influencing how much emphasis is
    # placed on accurately predicting bounding box coordinates
    # Default = 7.5
    "train_box_emph": 7.5,
    # Weight of the classification loss in the total loss function, affecting the importance
    # of correct class prediction relative to other components
    # Default = 0.5
    "train_class_emph": 0.5,

    # Training hyperparameters
    # Choice of optimizer for training. Options include SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc.
    # Set to 'auto' to let Ultralytics select the optimizer and its hyperparameters.
    # For this dataset and model, auto selects AdamW with lr0=0.002 and momentum=0.9.
    # Note: the values below for train_lr0 and train_momentum are ignored when
    # train_optimizer is set to 'auto'. They are documented here only to show the
    # effective values that the manuscript reports.
    "train_optimizer": 'auto',
    # Utilizes a cosine learning rate scheduler, adjusting the learning rate following a cosine curve over epochs
    # Default = False
    "train_cos_lr": True,
    # Initial learning rate.
    # Ignored when train_optimizer is 'auto'.
    # Effective value with auto mode for this setup: 0.002.
    "train_lr0": 0.002,
    # Final learning rate as a fraction of the initial rate = (lr0 * lrf), used in conjunction with
    # schedulers to adjust the learning rate over time
    # Default = 0.01
    "train_lrf": 0.01,
    # Momentum factor for SGD or beta1 for Adam optimizers.
    # Ignored when train_optimizer is 'auto'.
    # Effective value with auto mode for this setup: 0.9.
    "train_momentum": 0.9,
    # L2 regularization term, penalizing large weights to prevent overfitting
    # Default = 0.0005
    "train_weight_decay": 0.0005,
    # Number of epochs for learning rate warmup, gradually increasing the learning rate from a low
    # value to the initial learning rate to stabilize training early on
    # Default = 3.0
    "train_warmup_epochs": 5.0,
    # Initial momentum for warmup phase, gradually adjusting to the set momentum over the warmup period
    # Default = 0.8
    "train_warmup_momentum": 0.8,
    # Learning rate for bias parameters during the warmup phase, helping stabilize model training in the initial epochs
    # Default = 0.1
    "train_warmup_bias_lr": 0.1,
    # Dropout rate for regularization in classification tasks, preventing overfitting by randomly omitting units during training
    # Default = 0.0
    "train_dropout": 0.0,

    # Data augmentation
    # https://github.com/orgs/ultralytics/discussions/4142
    # https://docs.ultralytics.com/usage/cfg/#augmentation-settings
    # Adjust the hue of the image by a fraction of the color wheel, introducing color variability.
    # Helps the model generalize across different lighting conditions.
    # Default = 0.015
    "aug_hsv_h": 0.015,
    # Alters the saturation of the image by a fraction, affecting the intensity of colors.
    # Useful for simulating different environmental conditions.
    # Default = 0.7
    "aug_hsv_s": 0.3,
    # Modifies the value (brightness) of the image by a fraction, helping the model to perform
    # well under various lighting conditions.
    # Default = 0.4
    "aug_hsv_v": 0.2,
    # Rotate the image by a certain degree to simulate different orientations
    # Default = 0.0
    "aug_degrees": 10.0,
    # Translate the image horizontally and vertically to simulate different positions
    # Default = 0.1
    "aug_translate": 0.1,
    # Scale the image to simulate different sizes of objects
    # Default = 0.5
    "aug_scale": 0.3,
    # Shear the image to simulate perspective changes
    # Default = 0.0
    "aug_shear": 0.0,
    # Adjust the perspective of the image
    # Default = 0.0
    "aug_perspective": 0.0,
    # Flip the image upside down or left to right
    # Default = 0.0
    "aug_flipud": 0.0,
    "aug_fliplr": 0.0,
    # Combine four training images into one to simulate different contexts
    # Default = 1.0
    "aug_mosaic": 1.0,
    # Overlay two images to create a single composite image
    # Default = 0.0
    "aug_mixup": 0.0,

    ####################
    # OBJECT DETECTION #
    ####################

    # Set to True if a pretrained yolov8 model is used
    # Set to False for a custom/self trained model
    "od_use_pretrained_model": False,
    # Size of the pretrained model (n, s, m, l, x)
    "od_pretrained_model_size": "m",
    # Name of custom model, in case 'od_use_pretrained_model' is set to False
    # For the nuclei model: nuclei_DIC_best.pt
    # For the filopodia model: filopodia_DIC_best.pt
    "od_custom_model_name": "filopodia_DIC_best.pt",

    # https://docs.ultralytics.com/usage/cfg/#predict-settings
    # Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned
    # Empty list = all classes
    "od_class_selection": [],
    # Maximum number of detections allowed per image. Limits the total number of objects the model
    # can detect in a single inference, preventing excessive outputs in dense scenes (default 300)
    "od_max_detections": 500,
    # Intersection Over Union (IoU) threshold:
    # Lower values result in fewer detections by eliminating overlapping boxes,
    # useful for reducing duplicates
    # Default: 0.7
    "od_iou": 0.8,
    # Sets the minimum confidence threshold for detections. Objects detected with confidence below
    # this threshold will be disregarded
    # Default: 0.4
    "od_min_conf": 0.4,
    # agnostic_nms: avoid duplicate detections when you have overlapping objects of different classes
    "od_agnostic_nms": True,

    # Settings for bounding box annotations
    # Show labels AND confidence on top of the bounding boxes
    "od_show_labels": False,
    # Show bounding boxes
    "od_show_bbox": True,
    # Bounding box line thickness
    "od_bbox_line_thickness": 1,
    # Labels and confidence text thickness
    "od_bbox_text_thickness": 1,
    # Labels and confidence text size
    "od_bbox_text_scale": 0.4,

    # Show predicted images after prediction
    # Press any key to close image
    "od_show_predicted_images": False,
    # Set to true if predicted images are supposed to be saved
    "od_save_predicted_images": False,
    # Defines the image size for inference (w, h)
    "od_inf_size_img": (512, 512),
    # Set to true if images for prediction are rectangular,
    # Set to false if images are squared
    "od_rectangular_img": False,
    # If this is set to true, a text file with the detection
    # results will be saved in the output folder
    "od_export_results": True,
    # Name of the summary result file
    "od_export_file_name": "results.txt",
    # Save bounding box results from predictions as txt file
    "od_save_bb_results": True,
    # Save bounding box results from predictions as a YOLO-format txt file per image
    # (class_id x_center y_center width height, normalized 0-1).
    # If False, only the summary results.txt is written
    "od_save_bbox_txt": True,

    ### RESULT WINDOW (IMAGES) ###
    # Result window title
    "window_results_title": "YOLO Detection: Results",
    # Result window position from the left upper corner in px
    "window_results_x_pos": 100,
    "window_results_y_pos": 100,

    #################
    # CLASS COUNTER #
    #################

    # Show class counter
    "od_show_class_counter": False,
    # x,y-coordinates of the class counter from the left upper corner in px
    "od_counter_offset": (10, 40),
    # Line height, depends on font size (scale)
    "od_counter_line_height": 25,
    # Font size (determines line height)
    "od_counter_scale": 0.7,
    # Class counter text color
    # In OpenCV BGR format -> white
    "od_counter_color": (255, 255, 255),
    # Class counter text thickness
    "od_counter_thickness": 1,

    ##################
    # IMAGE SPLITTER #
    ##################

    # Number of datasets to create (1-999)
    "split_num_datasets": 3,
    # Extension of images to split
    "split_img_extension": ".png",
    # Validation split
    "split_is_val_split": True,
    # Fraction of images for validation (0.0 - 1.0)
    "split_val_split": 0.2,
    # Test split
    "split_is_test_split": False,
    # Fraction of images for testing (0.0 - 1.0)
    "split_test_split": 0.2,

    #########
    # PATHS #
    #########

    "pth_models": "models/",
    "pth_yolo_models": "models/yolo_models/",
    "pth_custom_models": "models/custom_models/",
    "pth_output": "output/",
    "pth_predictions": "predictions/",
    "pth_dataset_info": "train/",
    "pth_training_images": "train/images/train/",
    "pth_validation_images": "train/images/val/",
    "pth_training_labels": "train/labels/train/",
    "pth_validation_labels": "train/labels/val/",
    "pth_splitin_data_img": "img_splitter/input/images/",
    "pth_splitin_data_label": "img_splitter/input/labels/",
    "pth_splitout": "img_splitter/output/",
}