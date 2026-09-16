# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

##################################
# Predict on images in a folder  #
##################################

# ===== Own Modules =====
from image import ImageOD


class Predict():

    #############################################################################################################
    # CONSTRUCTOR:

    def __init__(self) -> None:
        pass

    #############################################################################################################
    # CALL:

    def __call__(self) -> None:
        # Create image detection object and run prediction
        image_od = ImageOD()
        image_od()