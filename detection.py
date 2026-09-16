# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

####################
# Object detection #
####################

# ===== Third-Party Imports =====
import numpy as np
import supervision as sv
# ===== Own Modules =====
from counter import Count
from model import ModelOD
from settings import setting


class Detect():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self, model_object: ModelOD) -> None:

        # Image size for inference
        self.inf_img_size: tuple = setting["od_inf_size_img"]
        self.rectangular: bool = setting["od_rectangular_img"]

        # Load model
        self.model_object: ModelOD = model_object
        self.class_names: dict = self.model_object.class_names
        self.model = self.model_object.model

        # Prediction parameters
        # List of classes to detect, empty list = all classes
        if(len(setting["od_class_selection"]) == 0):
            self.od_class_list: list | None = None
        else:
            self.od_class_list = setting["od_class_selection"]
        # Sets the minimum confidence threshold for detections
        self.min_conf: float = setting["od_min_conf"]
        # Limits the amount of detections on an image (default 300)
        self.max_detections: int = setting["od_max_detections"]
        # Intersection Over Union (IoU) threshold
        self.iou: float = setting["od_iou"]
        # agnostic_nms: Avoid duplicate bounding boxes for the same object
        self.agnostic_nms: bool = setting["od_agnostic_nms"]

        # Class counter object
        self.show_class_counter: bool = setting["od_show_class_counter"]
        if(self.show_class_counter):
            self.class_counter: Count = Count()

        # Annotations
        self.show_labels: bool = setting["od_show_labels"]
        self.show_bbox: bool = setting["od_show_bbox"]
        # Object for box annotations
        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=setting["od_bbox_line_thickness"],
        )
        # Object for label annotations
        self.label_annotator = sv.LabelAnnotator(
            text_color=sv.Color.BLACK,
            text_thickness=setting["od_bbox_text_thickness"],
            text_scale=setting["od_bbox_text_scale"]
        )

    #############################################################################################################
    # METHODS:

    # Predict on images.
    # https://docs.ultralytics.com/modes/predict/
    # Args:
    #   source (np.ndarray): Image to predict on
    # Returns:
    #   list: Ultralytics results list
    def predict(self, source: np.ndarray) -> list:
        results = self.model.predict(
            source=source,
            agnostic_nms=self.agnostic_nms,
            conf=self.min_conf,
            iou=self.iou,
            imgsz=self.inf_img_size,
            rect=self.rectangular,
            max_det=self.max_detections,
            classes=self.od_class_list,
        )
        return results

    # Read detections from an Ultralytics result.
    # Args:
    #   result: Ultralytics result object
    # Returns:
    #   sv.Detections: Detections object
    def read_detections(self, result) -> sv.Detections:
        return sv.Detections.from_ultralytics(result)

    # Draw bounding boxes and labels on an image.
    # Args:
    #   img (np.ndarray): Input image
    #   detections (sv.Detections): Detections to annotate
    # Returns:
    #   np.ndarray: Annotated image
    def annotate_bboxes(self, img: np.ndarray, detections: sv.Detections) -> np.ndarray:
        # Format custom labels
        labels = [f"{self.class_names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _, _
            in detections]

        # Annotate bounding boxes
        if(self.show_bbox):
            img = self.box_annotator.annotate(scene=img.copy(), detections=detections)
        # Annotate labels
        if(self.show_labels):
            img = self.label_annotator.annotate(scene=img, detections=detections, labels=labels)

        return img

    # Format the detection results for one image as a dict.
    # Args:
    #   result: Ultralytics result object
    # Returns:
    #   dict: Class name to count mapping
    def get_detection_result(self, result) -> dict:
        # Detection results as dict
        det_result: dict = {}
        # Read detections per image and class
        # Check first if there are any detections in the image
        if(len(result.boxes)):
            # Iterate over class names
            for class_index, class_name in self.class_names.items():
                # Determine count per class
                object_count = result.boxes.cls.tolist().count(class_index)
                # Build dict with results (class: count)
                det_result[class_name] = object_count
        # If there are no detections
        else:
            # Iterate over class names
            for class_index, class_name in self.class_names.items():
                # Detections for class is 0
                det_result[class_name] = 0

        return det_result

    # Write bounding boxes for one image to a YOLO-format text file.
    # Format per line: class_id x_center y_center width height (all normalized 0-1).
    # Args:
    #   result: Ultralytics result object
    #   output_pth (str): Path to the output text file
    # Returns:
    #   None
    def write_bbox_txt(self, result, output_pth: str) -> None:
        img_h, img_w = result.orig_shape
        with open(output_pth, "w") as f:
            for box in result.boxes:
                # xyxy (top-left, bottom-right) in pixel coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # Convert to YOLO format: x_center, y_center, width, height (normalized)
                x_center = ((x1 + x2) / 2) / img_w
                y_center = ((y1 + y2) / 2) / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h
                class_id = int(box.cls[0])
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    #############################################################################################################
    # CALL:

    def __call__(self, img: np.ndarray) -> tuple:
        # Predict objects in image
        results = self.predict(img)

        # Format results as dict
        result_dict = self.get_detection_result(results[0])

        # Read detections from image
        detections = self.read_detections(results[0])

        # Draw bounding boxes with labels
        img = self.annotate_bboxes(img, detections)

        # Show class counter
        if(self.show_class_counter):
            img = self.class_counter(results[0], img, self.class_names)

        return img, result_dict, results[0]