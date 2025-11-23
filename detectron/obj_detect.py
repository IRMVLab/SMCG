import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings

# import cv2
import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from .predictor import VisualizationDemo

# constants
# WINDOW_NAME = "COCO detections"
config_file = "./detectron/config/faster_rcnn_R_50_FPN_3x.yaml"
is_cpu = "cpu"
if torch.cuda.is_available():
    is_cpu = "cuda"
opts = [
    "MODEL.WEIGHTS",
    "./detectron/model/model_final_280758.pkl",
    "MODEL.DEVICE",
    is_cpu,
]
confidence_threshold = 0.5


class ObjDetect:
    def __init__(self):
        mp.set_start_method("spawn", force=True)
        setup_logger(name="fvcore")
        self.logger = setup_logger()

        cfg = self.setup_cfg()
        self.demo = VisualizationDemo(cfg)
        print("ObjDetect init success")

    def setup_cfg(self):
        # load config from file and command-line arguments
        cfg = get_cfg()
        # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
        # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
        # add_panoptic_deeplab_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(opts)
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
            confidence_threshold
        )
        cfg.freeze()
        return cfg

    def get_parser(self):
        parser = argparse.ArgumentParser(
            description="Detectron2 demo for builtin configs"
        )
        parser.add_argument(
            "--config-file",
            default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument(
            "--webcam", action="store_true", help="Take inputs from webcam."
        )
        parser.add_argument("--video-input", help="Path to video file.")
        parser.add_argument(
            "--input",
            nargs="+",
            help="A list of space separated input images; "
            "or a single glob pattern such as 'directory/*.jpg'",
        )
        parser.add_argument(
            "--output",
            help="A file or directory to save output visualizations. "
            "If not given, will show output in an OpenCV window.",
        )

        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum score for instance predictions to be shown",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )
        return parser

    # def test_opencv_video_format(self, codec, file_ext):
    #     with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
    #         filename = os.path.join(dir, "test_file" + file_ext)
    #         writer = cv2.VideoWriter(
    #             filename=filename,
    #             fourcc=cv2.VideoWriter_fourcc(*codec),
    #             fps=float(30),
    #             frameSize=(10, 10),
    #             isColor=True,
    #         )
    #         [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
    #         writer.release()
    #         if os.path.isfile(filename):
    #             return True
    #         return False

    def obj_detect(self, rgb_image):
        objects, obj_scores, obj_classes, img = self.demo.run_on_image(rgb_image)
        return [objects, obj_scores, obj_classes], img
