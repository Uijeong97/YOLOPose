import json
import argparse
import os
import numpy as np
import pandas as pd
import sys

from revised_ftns import revised_people_label, concat_json_and_txt
from darknet_to_coco import dark_to_coco

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--image_dir", type=str, default=r"C:\Users\soma\SMART_Referee\SMART_Referee_DL\data\COCO_human",
                    help="The path of the input image.")
parser.add_argument("--json_dir", type=str, default=r"C:\Users\soma\SMART_Referee\SMART_Referee_DL\data\COCO_human_json",
                    help="The path of the json file.")
parser.add_argument("--revised_label_dir", type=str, default=r"C:\Users\soma\SMART_Referee\SMART_Referee_DL\data\COCO_revised_label",
                    help="The path of the revised label data.")
parser.add_argument("--training_txt_dir", type=str, default=r"C:\Users\soma\YOLOPose\Pose_estimate\data\my_data",
                    help="The path of the training data.")
parser.add_argument("--training_txt_name", type=str, default=r"train.txt",
                    help="The name of the training data.")
args = parser.parse_args()


# Revise people label 0 to 18
revised_people_label(args.image_dir, args.revised_label_dir)

# concat json data to coco person data
concat_json_and_txt(args.json_dir, args.revised_label_dir)

# make training file
dark_to_coco(args.revised_label_dir, args.training_txt_dir, args.training_txt_name, args.image_dir)