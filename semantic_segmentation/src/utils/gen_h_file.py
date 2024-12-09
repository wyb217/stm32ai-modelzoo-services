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
import onnxruntime

from cfg_utils import aspect_ratio_dict
from cfg_utils import color_mode_n6_dict

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
    if os.path.basename(quantized_model_path).endswith(".tflite"):
        interpreter_quant = tf.lite.Interpreter(model_path=quantized_model_path)
        input_details = interpreter_quant.get_input_details()[0]
        output_details = interpreter_quant.get_output_details()[0]
        input_shape = input_details['shape']
    elif os.path.basename(quantized_model_path).endswith(".onnx"):
        session = onnxruntime.InferenceSession(quantized_model_path, None)
        model_input_shape = session.get_inputs()[0].shape
        input_shape = model_input_shape[-3:]

    class_names = params.dataset.class_names

    path = os.path.join(HydraConfig.get().runtime.output_dir, "C_header/")

    try:
        os.mkdir(path)
    except OSError as error:
        print(error)

    TFLite_Detection_PostProcess_id = False
    XY, WH = None, None


    classes = '{\\\n'

    for i, x in enumerate(params.dataset.class_names):
        if i == (len(class_names) - 1):
            classes = classes + '   "' + str(x) + '"' + '}\\'
        else:
            classes = classes + '   "' + str(x) + '"' + ' ,' + ('\\\n' if (i % 5 == 0 and i != 0) else '')

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
        f.write("#define CAMERA_FLIP CMW_MIRRORFLIP_NONE\n")
        f.write("\n\n")
        f.write("#define ASPECT_RATIO_CROP       (2)\n")
        f.write("#define ASPECT_RATIO_FIT        (1)\n")
        f.write("#define ASPECT_RATIO_FULLSCREEN (3)\n")
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

        if params.general.model_type == "deeplab_v3":
            f.write("#define POSTPROCESS_TYPE    POSTPROCESS_SSEG_DEEPLAB_V3_UF\n\n")
        else:
            raise TypeError("please select one of this supported post processing options [ deeplab_v3 ]")


        f.write("#define NN_HEIGHT     ({})\n".format(int(input_shape[1])))
        f.write("#define NN_WIDTH      ({})\n".format(int(input_shape[2])))
        f.write("#define NN_BPP 3")
        f.write("\n\n")
        f.write("#define COLOR_BGR (0)\n")
        f.write("#define COLOR_RGB (1)\n")
        f.write("#define COLOR_MODE    {}\n".format(color_mode_n6_dict[params.preprocessing.color_mode]))

        f.write("/* Classes */\n")
        f.write("/* I/O configuration */\n")
        f.write("#define NB_CLASSES   ({})\n".format(len(class_names)))
        f.write("#define CLASSES_TABLE const char* classes_table[NB_CLASSES] = {}\n".format(classes))

        if (params.general.model_type == "deeplab_v3"):
            f.write("/* Postprocessing ST_SSD configuration */\n")
            f.write("#define AI_SSEG_DEEPLABV3_PP_NB_CLASSES         ({})\n".format(len(class_names)))
            f.write("#define AI_SSEG_DEEPLABV3_PP_WIDTH              (NN_WIDTH)\n")
            f.write("#define AI_SSEG_DEEPLABV3_PP_HEIGHT             (NN_HEIGHT)\n\n")

        f.write('#define WELCOME_MSG_1     "{}"\n'.format(os.path.basename(params.general.model_path)))
        # @Todo retieve info from stedgeai output
        f.write('#define WELCOME_MSG_2       "{}"\n'.format("Model Running in STM32 MCU internal memory"))

        f.write("\n")
        f.write("#endif      /* APP_CONFIG */\n")

    return TFLite_Detection_PostProcess_id, quantized_model_path