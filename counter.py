# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

##########################
# Onscreen Class Counter #
##########################

# ===== Third-Party Imports =====
import numpy as np
# ===== Own Modules =====
import functions as fcn
from settings import setting


class Count():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self) -> None:
        self.counter_offset = setting['od_counter_offset']
        self.counter_line_height = setting['od_counter_line_height']
        self.counter_scale = setting["od_counter_scale"]
        self.counter_color = setting["od_counter_color"]
        self.counter_thickness = setting["od_counter_thickness"]

    #############################################################################################################
    # CALL:

    # Print object counter onscreen while object detection.
    # https://github.com/ultralytics/yolov5/issues/12947
    # Args:
    #   result: Ultralytics result object
    #   img (np.ndarray): Image to annotate
    #   class_names (dict): Class index to name mapping
    # Returns:
    #   np.ndarray: Annotated image
    def __call__(self, result, img: np.ndarray, class_names: dict) -> np.ndarray:

        # Display header
        pos = (self.counter_offset[0], self.counter_offset[1] + self.counter_line_height)
        fcn.annotate_text(
            img,
            "COUNTS:",
            pos,
            self.counter_scale,
            self.counter_color,
            self.counter_thickness,
        )

        # Count lines for line height
        line_count = 2
        # Read detections per image and class
        # Check first if there are any detections in the image
        if(len(result.boxes)):
            # Iterate over class names
            for class_index, class_name in class_names.items():
                # Determine count per class
                object_count = result.boxes.cls.tolist().count(class_index)
                # Only add class if count > 0
                if(object_count > 0):
                    label = f"{class_name}: {object_count}"
                    # Display lines
                    pos = (self.counter_offset[0], self.counter_offset[1] + (line_count * self.counter_line_height))
                    fcn.annotate_text(
                        img,
                        label,
                        pos,
                        self.counter_scale,
                        self.counter_color,
                        self.counter_thickness,
                    )
                    # Increment line count
                    line_count += 1

        # In case there are no detections in the image
        else:
            pos = (self.counter_offset[0], self.counter_offset[1] + (line_count * self.counter_line_height))
            fcn.annotate_text(
                img,
                "no detections",
                pos,
                self.counter_scale,
                self.counter_color,
                self.counter_thickness,
            )

        return img