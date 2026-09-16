# Diffusion-Based-Phenotypic-Extrapolation-YOLO
# Copyright (C) 2026 Markus Reichold <markus.reichold@ur.de>
# SPDX-License-Identifier: AGPL-3.0-or-later

#####################
# Main entry point  #
#####################

# ===== Own Modules =====
import functions as fcn
from settings import setting
from train import Train
from predict import Predict
from img_split import Img_Split


def main() -> None:

    # Show CUDA status and software versions
    fcn.show_cuda_and_versions()
    # Create program folders which are ignored by git
    fcn.create_prg_folders()

    #############
    # Main Menu #
    #############

    while(True):
        print("\n:MAIN MENU:")
        print("  1) Train model")
        print("  2) Predict on images")
        print("  3) Dataset splitter")
        print("  4) Exit program")
        menu = fcn.input_int("Please choose: ")

        #################
        # Train Network #
        #################

        if(menu == 1):
            print("\n:TRAIN NETWORK:")
            od_train = Train()
            od_train()

        ###################
        # Predict Images  #
        ###################

        elif(menu == 2):
            print("\n:PREDICTION ON IMAGES:")
            od_predict = Predict()
            od_predict()

        #####################
        # Dataset Splitter  #
        #####################

        elif(menu == 3):
            print("\n:DATASET SPLITTER:")
            img_split = Img_Split()
            img_split()

        ################
        # Exit Program #
        ################

        elif(menu == 4):
            print("\nExit program...")
            break

        # Wrong input
        else:
            print("Not a valid option!")


if __name__ == "__main__":
    main()