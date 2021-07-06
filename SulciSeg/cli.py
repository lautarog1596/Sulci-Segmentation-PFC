import sys
import os
from os.path import dirname
import time
cwd = os.getcwd()
sys.path.append(dirname(cwd))

from PostProcessor.Postprocessor.defuse import defuse_cli
from PostProcessor.Postprocessor.report import report_cli
from TrainTestModel.Unet.config import load_config
from TrainTestModel.predict import predict_cli
from TrainTestModel.train import train_cli
from PreProcessor.preprocess import zscore_norm_cli


def print_menu():
    print("\n========= SulciSeg cli ==========\n"
          "Choose an option and press enter \n"
          "  1 - Normalize \n"
          # "  2 - Train \n"
          "  2 - Predict and defuse \n"
          "  3 - Generate report \n"
          "  4 - Exit \n"
          "=================================")


def normalize():
    print("Enter the path of the image (s) and press enter.")
    path = ""
    while not os.path.exists(path):
        path = input()
        if os.path.exists(path):  # if path exist
            print("Normalizing...")
            t = time.time()
            path_out = zscore_norm_cli(path)
            elapsed = time.time() - t
            print(f"Normalized images saved in: {path_out}")
            print(f"*Elapsed: {elapsed} seconds\n")
            return
        else:
            print("Invalid path. Try again.")


def predict_and_defuse():
    print("Enter the path of the image (s) and press enter.")
    path = ""
    while not os.path.exists(path):
        path = input()
        if os.path.exists(path):  # if path exist
            print("Predicting...")
            t = time.time()
            path_out_predict = predict_cli(path)
            elapsed = time.time() - t
            print(f"*Elapsed: {elapsed} seconds")
            print("Defusing and relabeling...")
            t = time.time()
            path_out_defused = defuse_cli(path_out_predict) # /home/lau/Escritorio/predictions/pred_M26_whole_2.nii.gz
            elapsed = time.time() - t
            print(f"Predictions saved in: {path_out_defused}")
            print(f"*Elapsed: {elapsed} seconds\n")
            return
        else:
            print("Invalid path. Try again.")


# def train():
#     print("Enter the path of the training settings (train_parameters.yaml)")
#     path = ""
#     while not os.path.exists(path):
#         path = input()
#         if os.path.exists(path):  # if path exist
#             print("Loading training settings from file:")
#             path_yaml = dirname(cwd) + '/TrainTestModel/train_parameters.yaml'
#             config = load_config(path_yaml)
#             print(path_yaml)
#             train_cli(config)
#             return
#         else:
#             print("Invalid path. Try again.")


def report():
    print("Enter the path of the prediction image (s) and press enter.")
    path = ""
    while not os.path.exists(path):
        path = input()
        if os.path.exists(path):  # if path exist
            print("Enter the path of the brain image (s) and press enter.")
            path_brain = ""
            while not os.path.exists(path_brain):
                path_brain = input()
                if os.path.exists(path_brain):  # if path exist
                    print("Generating...")
                    t = time.time()
                    path_out = report_cli(path, path_brain)
                    elapsed = time.time() - t
                    print(f"Report generated in: {path_out}")
                    print(f"*Elapsed: {elapsed} seconds\n")
                    return
                else:
                    print("Invalid path. Try again.")
        else:
            print("Invalid path. Try again.")


def menu():
    print_menu()
    user_input = 0

    while user_input != 4:
        user_input = int(input())
        if user_input == 1:
            normalize()
            print_menu()
        # elif user_input == 2:
        #     train()
        #     print_menu()
        elif user_input == 2:
            predict_and_defuse()
            print_menu()
        elif user_input == 3:
            report()
            print_menu()
        elif user_input == 4:
            print("Exiting...")
        else:
            print("Invalid option")


if __name__ == "__main__":
    menu()
