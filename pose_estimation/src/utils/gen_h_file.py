# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import os
import glob
import re


from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np
import tensorflow as tf

from cfg_utils import color_mode_n6_dict
from cfg_utils import aspect_ratio_dict


def gen_h_user_file_n6(config: DictConfig = None, quantized_model_path: str = None) -> None:
    """
    Generates a C header file containing user configuration for the AI model.

    Args:
        config: A configuration object containing user configuration for the AI model.
        quantized_model_path: The path to the quantized model file.

    """
    class Flags:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    params = Flags(**config)
    interpreter_quant = tf.lite.Interpreter(model_path=quantized_model_path)
    input_details = interpreter_quant.get_input_details()[0]
    output_details = interpreter_quant.get_output_details()[0]
    input_shape = input_details['shape']

    path = os.path.join(HydraConfig.get().runtime.output_dir, "C_header/")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    TFLite_Detection_PostProcess_id = False

    with open(os.path.join(path, "app_config.h"), "wt") as f:
        f.write("/**\n")
        f.write("******************************************************************************\n")
        f.write("* @file    app_config.h\n")
        f.write("* @author  GPM Application Team\n")
        f.write("*\n")
        f.write("******************************************************************************\n")
        f.write("* @attention\n")
        f.write("*\n")
        f.write("* Copyright (c) 2023 STMicroelectronics.\n")
        f.write("* All rights reserved.\n")
        f.write("*\n")
        f.write("* This software is licensed under terms that can be found in the LICENSE file\n")
        f.write("* in the root directory of this software component.\n")
        f.write("* If no LICENSE file comes with this software, it is provided AS-IS.\n")
        f.write("*\n")
        f.write("******************************************************************************\n")
        f.write("*/\n\n")
        f.write("/* ---------------    Generated code    ----------------- */\n")
        f.write("#ifndef APP_CONFIG\n")
        f.write("#define APP_CONFIG\n\n")
        f.write('#include "arm_math.h"\n\n')
        f.write("#define USE_DCACHE\n\n")
        f.write("/*Defines: CMW_MIRRORFLIP_NONE; CMW_MIRRORFLIP_FLIP; CMW_MIRRORFLIP_MIRROR; CMW_MIRRORFLIP_FLIP_MIRROR;*/\n")
        f.write("#define CAMERA_FLIP CMW_MIRRORFLIP_NONE")
        f.write("\n\n")
        f.write("#define ASPECT_RATIO_CROP (1) /* Crop both pipes to nn input aspect ratio; Original aspect ratio kept */\n")
        f.write("#define ASPECT_RATIO_FIT (2) /* Resize both pipe to NN input aspect ratio; Original aspect ratio not kept */\n")
        f.write("#define ASPECT_RATIO_FULLSCREEN (3) /* Resize camera image to NN input size and display a fullscreen image */\n")
        f.write("\n")
        f.write("#define ASPECT_RATIO_MODE    {}\n".format(aspect_ratio_dict[params.preprocessing.resizing.aspect_ratio]))
        f.write("\n")

        f.write("#define CAPTURE_FORMAT DCMIPP_PIXEL_PACKER_FORMAT_RGB565_1\n")
        f.write("#define CAPTURE_BPP 2\n")
        f.write("/* Leave the driver use the default resolution */\n")
        f.write("#define CAMERA_WIDTH 0\n")
        f.write("#define CAMERA_HEIGHT 0\n\n")

        f.write("#define LCD_FG_WIDTH             800\n")
        f.write("#define LCD_FG_HEIGHT            480\n")
        f.write("#define LCD_FG_FRAMEBUFFER_SIZE  (LCD_FG_WIDTH * LCD_FG_HEIGHT * 2)\n")

        f.write("/* Postprocessing type configuration */\n")

        if params.general.model_type == "heatmaps_spe":
            f.write("#define POSTPROCESS_TYPE    POSTPROCESS_SPE_MOVENET_UF\n\n")
        elif params.general.model_type == "yolo_mpe":
            f.write("#define POSTPROCESS_TYPE    POSTPROCESS_MPE_YOLO_V8_UF\n\n")
        else:
            raise TypeError("please select one supported post processing options: heatmaps_spe or yolo_mpe")

        f.write("#define NN_HEIGHT     ({})\n".format(int(input_shape[1])))
        f.write("#define NN_WIDTH      ({})\n".format(int(input_shape[2])))
        f.write("#define NN_BPP 3")
        f.write("\n\n")
        f.write("#define COLOR_BGR (0)\n")
        f.write("#define COLOR_RGB (1)\n")
        f.write("#define COLOR_MODE    {}\n".format(color_mode_n6_dict[params.preprocessing.color_mode]))

        if params.general.model_type == "heatmaps_spe":
            f.write("\n/* Post processing values */\n")
            f.write("#define AI_POSE_PP_CONF_THRESHOLD              ({})\n".format(float(params.postprocessing.confidence_thresh)))
            f.write("#define AI_POSE_PP_POSE_KEYPOINTS_NB           ({})\n".format(int(output_details['shape'][3])))
            f.write("#define AI_SPE_MOVENET_POSTPROC_HEATMAP_WIDTH        (NN_WIDTH/4)\n")
            f.write("#define AI_SPE_MOVENET_POSTPROC_HEATMAP_HEIGHT       (NN_HEIGHT/4)\n")
            f.write("#define AI_SPE_MOVENET_POSTPROC_NB_KEYPOINTS         (AI_POSE_PP_POSE_KEYPOINTS_NB)        /* Only 13 and 17 keypoints are supported for the skeleton reconstruction */\n\n")
        elif params.general.model_type == "yolo_mpe":
            out_shape = output_details["shape"]
            nb_kpt = (out_shape[1]-5)/3
            f.write("\n/* Post processing values */\n")
            f.write("#define AI_MPE_YOLOV8_PP_NB_CLASSES (1)\n")
            f.write("#define AI_MPE_YOLOV8_PP_TOTAL_BOXES ({})\n".format(int(out_shape[2])))
            f.write("#define AI_MPE_YOLOV8_PP_MAX_BOXES_LIMIT ({})\n".format(int(params.postprocessing.max_detection_boxes)))
            f.write("#define AI_MPE_YOLOV8_PP_IOU_THRESHOLD ({})\n".format(float(params.postprocessing.NMS_thresh)))
            f.write("#define AI_MPE_YOLOV8_PP_CONF_THRESHOLD ({})\n\n".format(float(params.postprocessing.confidence_thresh)))

            f.write("#define AI_POSE_PP_CONF_THRESHOLD              (AI_MPE_YOLOV8_PP_CONF_THRESHOLD)\n")
            f.write("#define AI_POSE_PP_POSE_KEYPOINTS_NB           ({})\n".format(int(nb_kpt)))
        else:
            raise ValueError("model_type not supported")
        f.write('/* Display */\n')
        f.write('#define WELCOME_MSG_0       "Single/multi pose estimation - Hand landmark"\n')
        f.write('#define WELCOME_MSG_1       "{}"\n'.format(os.path.basename(params.general.model_path)))
        # @Todo retieve info from stedgeai output
        f.write('#define WELCOME_MSG_2       "{}"\n'.format("Model Running in STM32 MCU internal memory"))
        f.write("\n")
        f.write("#endif      /* APP_CONFIG */\n")

    return TFLite_Detection_PostProcess_id, quantized_model_path