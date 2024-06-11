# -*- coding: utf-8 -*-
"""
 @File    : hand_written.py
 @Time    : 2020/9/2 下午5:15
 @Author  : yizuotian
 @Description    :
"""

import argparse
import codecs
import glob
import multiprocessing as mp
import os
import time

import cv2
import numpy as np
import tqdm
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import VisualizationDemo
from keras.preprocessing.image import img_to_array

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def decode_recognition(rec):
    CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3',
                '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\',
                ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

    s = ''
    for c in rec:
        c = int(c)
        if c < 95:
            s += CTLABELS[c]
        elif c == 95:
            s += u'口'
    return s


def bezier_to_poly(bezier):
    # bezier to polygon
    u = np.linspace(0, 1, 20)
    bezier = bezier.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
    points = np.outer((1 - u) ** 3, bezier[:, 0]) \
             + np.outer(3 * u * ((1 - u) ** 2), bezier[:, 1]) \
             + np.outer(3 * (u ** 2) * (1 - u), bezier[:, 2]) \
             + np.outer(u ** 3, bezier[:, 3])
    points = np.concatenate((points[:, :2], points[:, 2:]), axis=0)

    return points


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
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


if __name__ == "__main__":
    """
    Usage:
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="1"

cd /home/mydir/pyspace/AdelaiDet
python demo/hand_written.py --config-file configs/BAText/TotalText/attn_R_50.yaml \
--input /home/mydir/dataset/handWriting/test \
--output /home/mydir/dataset/handWriting/test.vis \
--confidence-threshold 0.4 \
--opts MODEL.WEIGHTS /home/mydir/pretrained_model/tt_attn_R_50.pth \
MODEL.FCOS.NMS_TH 0.05 INPUT.MIN_SIZE_TEST 200 INPUT.MAX_SIZE_TEST 400 
    
    scp -rp hand_written.py root@m2:/home/mydir/pyspace/AdelaiDet/demo/
    
    scp -rp root@m2:/home/mydir/pyspace/AdelaiDet/key.csv ./
    """
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
            args.input = sorted(args.input, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        with codecs.open('./key.csv', mode='w', encoding='utf-8-sig') as w:
            # w.write(codecs.BOM_UTF8)
            for path in tqdm.tqdm(args.input, disable=not args.output):
                # use PIL, to be consistent with evaluation
                img = read_image(path, format="BGR")
                start_time = time.time()
                predictions, visualized_output = demo.run_on_image(img)
                logger.info(
                    "{}: detected {} instances in {:.2f}s".format(
                        path, len(predictions["instances"]), time.time() - start_time
                    )
                )
                instances = predictions["instances"].to(demo.cpu_device)
                recs = instances.recs
                texts = [decode_recognition(rec) for rec in recs]

                beziers = instances.beziers.numpy()
                xs = [bezier_to_poly(bezier)[0, 0] for bezier in beziers]

                texts_xs = [(text, x) for text, x in zip(texts, xs)
                            if not text.endswith('NOM') and not text == 'NOME']
                texts = [text.upper() for text, x in sorted(texts_xs, key=lambda text_x: text_x[1])]
                logger.info(texts)

                file_prefix = os.path.splitext(os.path.basename(path))[0]
                texts = ['DEFAULT'] if len(texts) == 0 else texts
                w.write('{},{}\n'.format(file_prefix, ' '.join(texts)))

                if args.output:
                    if os.path.isdir(args.output):
                        assert os.path.isdir(args.output), args.output
                        out_filename = os.path.join(args.output, os.path.basename(path))
                    else:
                        assert len(args.input) == 1, "Please specify a directory with args.output"
                        out_filename = args.output
                    visualized_output.save(out_filename)
                else:
                    cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                    if cv2.waitKey(0) == 27:
                        break  # esc to quit
