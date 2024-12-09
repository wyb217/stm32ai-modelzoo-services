#  /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig
from munch import DefaultMunch
import numpy as np
from hydra.core.hydra_config import HydraConfig

from cfg_utils import postprocess_config_dict, check_config_attributes, parse_tools_section, parse_benchmarking_section, \
                      parse_mlflow_section, parse_top_level, parse_general_section, parse_quantization_section, \
                      parse_training_section, parse_prediction_section, parse_deployment_section, check_hardware_type, \
                      parse_evaluation_section
from typing import Dict, List

    
def parse_dataset_section(cfg: DictConfig, mode: str = None,
                          mode_groups: DictConfig = None,
                          hardware_type: str = None) -> None:

    # cfg: dictionary containing the 'dataset' section of the configuration file
    # cfg: dictionary containing the 'dataset' section of the configuration file

    legal = ["name", "class_names", "training_path", "validation_path", "validation_split", "test_path",
             "quantization_path", "quantization_split", "seed"]

    required = []
    one_or_more = []
    if mode in mode_groups.training:
        required += ["training_path", ]
    elif mode in mode_groups.evaluation:
        one_or_more += ["training_path", "test_path"]
        
    if mode not in ("quantization", "benchmarking", "chain_qb"):
        if hardware_type == "MCU":
            required += ["class_names", ]
    
    check_config_attributes(cfg, specs={"legal": legal, "all": required, "one_or_more": one_or_more},
                            section="dataset")

    # Set default values of missing optional attributes
    if not cfg.name:
        cfg.name = "unnamed"
    if not cfg.validation_split:
        cfg.validation_split = 0.2
    cfg.seed = cfg.seed if cfg.seed else 123

    # Check the value of validation_split if it is set
    if cfg.validation_split:
        split = cfg.validation_split
        if split <= 0.0 or split >= 1.0:
            raise ValueError(f"\nThe value of `validation_split` should be > 0 and < 1. Received {split}\n"
                             "Please check the 'dataset' section of your configuration file.")

    # Check the value of quantization_split if it is set
    if cfg.quantization_split:
        split = cfg.quantization_split
        if split < 0 or split > 1:
            raise ValueError(f"\nThe value of `quantization_split` should be > 0 and < 1. Received {split}\n"
                             "Please check the 'dataset' section of your configuration file.")


def get_class_names_from_file(cfg: DictConfig) -> List:
    if cfg.deployment.label_file_path :
        with open(cfg.deployment.label_file_path, 'r') as file:
            class_names = [line.strip() for line in file]
    return class_names
    
    
def parse_preprocessing_section(cfg: DictConfig,
                                mode:str = None) -> None:
    # cfg: 'preprocessing' section of the configuration file
    legal = ["rescaling", "resizing", "color_mode"]
    if mode == 'deployment':
        # removing the obligation to have rescaling for the 'deployment' mode
        required=["resizing", "color_mode"]
        check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="preprocessing")
    else:
        required=legal
        check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="preprocessing")
        legal = ["scale", "offset"]
        check_config_attributes(cfg.rescaling, specs={"legal": legal, "all": legal}, section="preprocessing.rescaling")

    legal = ["interpolation", "aspect_ratio"]
    check_config_attributes(cfg.resizing, specs={"legal": legal, "all": legal}, section="preprocessing.resizing")
    if cfg.resizing.aspect_ratio not in ("fit", "crop", "padding"):
        raise ValueError("\nSupported methods for resizing images are 'fit', 'crop' and 'padding'. "
                         f"Received {cfg.resizing.aspect_ratio}\n"
                         "Please check the `resizing.aspect_ratio` attribute in "
                         "the 'preprocessing' section of your configuration file.")
                         
    # Check resizing interpolation value
    interpolation_methods = ["bilinear", "nearest", "area", "lanczos3", "lanczos5", "bicubic", "gaussian",
                             "mitchellcubic"]
    if cfg.resizing.interpolation not in interpolation_methods:
        raise ValueError(f"\nUnknown value for `interpolation` attribute. Received {cfg.resizing.interpolation}\n"
                         f"Supported values: {interpolation_methods}\n"
                         "Please check the 'resizing.attribute' in the 'preprocessing' section of your configuration file.")

    # Check color mode value
    color_modes = ["grayscale", "rgb", "rgba"]
    if cfg.color_mode not in color_modes:
        raise ValueError(f"\nUnknown value for `color_mode` attribute. Received {cfg.color_mode}\n"
                         f"Supported values: {color_modes}\n"
                         "Please check the 'preprocessing' section of your configuration file.")


def parse_data_augmentation_section(cfg: DictConfig) -> None:
    """
    This function checks the data augmentation section of the config file.
    The attribute that introduces the section is either `data_augmentation`
    or `custom_data_augmentation`. If it is `custom_data_augmentation`,
    the name of the data augmentation function that is provided must be
    different from `data_augmentation` as this is a reserved name.

    Arguments:
        cfg (DictConfig): The entire configuration file as a DefaultMunch dictionary.
        config_dict (Dict): The entire configuration file as a regular Python dictionary.

    Returns:
        None
    """

    if cfg.data_augmentation and cfg.custom_data_augmentation:
        raise ValueError("\nThe `data_augmentation` and `custom_data_augmentation` attributes "
                         "are mutually exclusive.\nPlease check your configuration file.")
    
    if cfg.data_augmentation:
        data_aug = DefaultMunch.fromDict({})
        # The name of the Model Zoo data augmentation function is 'data_augmentation'.
        data_aug.function_name = "data_augmentation"
        data_aug.config = deepcopy(cfg.data_augmentation)

    if cfg.custom_data_augmentation:
        check_attributes(cfg.custom_data_augmentation,
                         expected=["function_name"],
                         optional=["config"],
                         section="custom_data_augmentation")
        if cfg.custom_data_augmentation["function_name"] == "data_augmentation":
            raise ValueError("\nThe function name `data_augmentation` is reserved.\n"
                             "Please use another name (attribute `function_name` in "
                             "the 'custom_data_augmentation' section).")
                                                          
        data_aug = DefaultMunch.fromDict({})
        data_aug.function_name = cfg.custom_data_augmentation.function_name
        if cfg.custom_data_augmentation.config:
            data_aug.config = deepcopy(cfg.custom_data_augmentation.config)
    
    cfg.data_augmentation = data_aug


def parse_postprocessing_section(cfg: DictConfig, model_type: str) -> None:
    # cfg: 'postprocessing' section of the configuration file

    legal = ["confidence_thresh", "NMS_thresh", "IoU_eval_thresh", "yolo_anchors", "plot_metrics",
             'max_detection_boxes']
    required = ["confidence_thresh", "NMS_thresh", "IoU_eval_thresh"]
    check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="postprocessing")

    if model_type == "tiny_yolo_v2":
        cfg.network_stride = 32
    elif model_type == "st_yolo_lc_v1":
        cfg.network_stride = 16
    elif model_type == "st_yolo_x":
        cfg.network_stride = [8,16,32]
    
    # Set default YOLO anchors
    if not cfg.yolo_anchors and model_type == "st_yolo_x":
        cfg.yolo_anchors = [0.5, 0.5]
    elif not cfg.yolo_anchors:
        cfg.yolo_anchors = [0.076023, 0.258508, 0.163031, 0.413531, 0.234769, 0.702585, 0.427054, 0.715892, 0.748154, 0.857092]

    # Check the YOLO anchors syntax and convert to a numpy array
    if len(cfg.yolo_anchors) % 2 != 0:
        raise ValueError("\nThe Yolo anchors list should contain an even number of floats.\n"
                         "Please check the value of the 'postprocessing.yolo_anchors' attribute "
                         "in your configuration file.")
    num_anchors = int(len(cfg.yolo_anchors) / 2)
    anchors = np.array(cfg.yolo_anchors, dtype=np.float32)
    cfg.yolo_anchors = np.reshape(anchors, [num_anchors, 2])
    
    cfg.plot_metrics = cfg.plot_metrics if cfg.plot_metrics is not None else False


def get_config(config_data: DictConfig) -> DefaultMunch:
    """
    Converts the configuration data, performs some checks and reformats
    some sections so that they are easier to use later on.

    Args:
        config_data (DictConfig): dictionary containing the entire configuration file.

    Returns:
        DefaultMunch: The configuration object.
    """

    config_dict = OmegaConf.to_container(config_data)

    # Restore booleans, numerical expressions and tuples
    # Expand environment variables
    postprocess_config_dict(config_dict, replace_none_string=True)

    # Top level section parsing
    cfg = DefaultMunch.fromDict(config_dict)
    mode_groups = DefaultMunch.fromDict({
        "training": ["training", "chain_tqeb", "chain_tqe"],
        "evaluation": ["evaluation", "chain_tqeb", "chain_tqe", "chain_eqe", "chain_eqeb"],
        "quantization": ["quantization", "chain_tqeb", "chain_tqe", "chain_eqe",
                         "chain_qb", "chain_eqeb", "chain_qd"],
        "benchmarking": ["benchmarking", "chain_tqeb", "chain_qb", "chain_eqeb"],
        "deployment": ["deployment", "chain_qd"],
        "prediction": ["prediction"]
    })
    mode_choices = ["training", "evaluation", "deployment",
                    "quantization", "benchmarking", "chain_tqeb", "chain_tqe",
                    "chain_eqe", "chain_qb", "chain_eqeb", "chain_qd", "prediction"]
    legal = ["general", "operation_mode", "dataset", "preprocessing", "data_augmentation",
             "training", "postprocessing", "quantization", "evaluation", "prediction", "tools",
             "benchmarking", "deployment", "mlflow", "hydra"]
    parse_top_level(cfg, 
                    mode_groups=mode_groups,
                    mode_choices=mode_choices,
                    legal=legal)
    print(f"[INFO] : Running `{cfg.operation_mode}` operation mode")

    # General section parsing
    if not cfg.general:
        cfg.general = DefaultMunch.fromDict({})
    legal = ["project_name", "model_path", "logs_dir", "saved_models_dir", "deterministic_ops",
            "display_figures", "global_seed", "gpu_memory_limit", "model_type", "num_threads_tflite"]
    required = ["model_type"]
    parse_general_section(cfg.general, 
                          mode=cfg.operation_mode, 
                          mode_groups=mode_groups,
                          legal=legal,
                          required=required,
                          output_dir = HydraConfig.get().runtime.output_dir)
                          
    # Select hardware_type from yaml information
    check_hardware_type(cfg, mode_groups)
                        
    # Dataset section parsing
    if not cfg.dataset:
        cfg.dataset = DefaultMunch.fromDict({})
                    
    parse_dataset_section(cfg.dataset,
                          mode=cfg.operation_mode,
                          mode_groups=mode_groups,
                          hardware_type=cfg.hardware_type)
                          
    # Preprocessing section parsing
    parse_preprocessing_section(cfg.preprocessing,
                                mode=cfg.operation_mode)

    # Data augmentation section parsing
    if cfg.operation_mode in mode_groups.training:
        if cfg.data_augmentation or cfg.custom_data_augmentation:
            parse_data_augmentation_section(cfg)

    # Training section parsing
    if cfg.operation_mode in mode_groups.training:
        model_path_used = bool(cfg.general.model_path)
        model_type_used = bool(cfg.general.model_type)
        legal = ["model", "batch_size", "epochs", "optimizer", "dropout", "frozen_layers",
                "callbacks", "trained_model_path", "resume_training_from"]
        parse_training_section(cfg.training, 
                               model_path_used=model_path_used,
                               model_type_used=model_type_used,
                               legal=legal)

    # Postprocessing section parsing
    # This section is always needed except for benchmarking.
    if cfg.operation_mode in (mode_groups.training + mode_groups.evaluation +
                              mode_groups.quantization + mode_groups.deployment +
                              mode_groups.prediction):
        if cfg.hardware_type == "MCU":
            parse_postprocessing_section(cfg.postprocessing, cfg.general.model_type)
            
    # Quantization section parsing
    if cfg.operation_mode in mode_groups.quantization:
        legal = ["quantizer", "quantization_type", "quantization_input_type",
                "quantization_output_type", "granularity", "export_dir", "optimize","target_opset"]
        parse_quantization_section(cfg.quantization,
                                   legal=legal)

    # Evaluation section parsing
    if cfg.operation_mode in mode_groups.evaluation and "evaluation" in cfg:
        legal = ["gen_npy_input", "gen_npy_output", "npy_in_name", "npy_out_name","target"]
        parse_evaluation_section(cfg.evaluation,
                                 legal=legal)

    # Prediction section parsing
    if cfg.operation_mode == "prediction":
        parse_prediction_section(cfg.prediction)

    # Tools section parsing
    if cfg.operation_mode in (mode_groups.benchmarking + mode_groups.deployment):
        parse_tools_section(cfg.tools, 
                            cfg.operation_mode,
                            cfg.hardware_type)

    #For MPU, check if online benchmarking is activated
    if cfg.operation_mode in mode_groups.benchmarking:
        if "STM32MP" in cfg.benchmarking.board :
            if cfg.operation_mode == "benchmarking" and not(cfg.tools.stm32ai.on_cloud):
                print("Target selected for benchmark :", cfg.benchmarking.board)
                print("Offline benchmarking for MPU is not yet available please use online benchmarking")
                exit(1)

    # Benchmarking section parsing
    if cfg.operation_mode in mode_groups.benchmarking:
        parse_benchmarking_section(cfg.benchmarking)
        if cfg.hardware_type == "MPU" :
            if not (cfg.tools.stm32ai.on_cloud):
                print("Target selected for benchmark :", cfg.benchmarking.board)
                print("Offline benchmarking for MPU is not yet available please use online benchmarking")
                exit(1)

    # Deployment section parsing
    if cfg.operation_mode in mode_groups.deployment:
        if cfg.hardware_type == "MCU":
            legal = ["c_project_path", "IDE", "verbosity", "hardware_setup"]
            legal_hw = ["serie", "board", "stlink_serial_number"]
            # Append additional items if hardware_type is "MCU_H7"
            if cfg.deployment.hardware_setup.serie == "STM32H7":
                legal_hw += ["input", "output"]
        else:
            legal = ["c_project_path", "label_file_path","board_deploy_path", "verbosity", "hardware_setup"]
            legal_hw = ["serie", "board", "ip_address"]
            if cfg.preprocessing.color_mode != "rgb":
                raise ValueError("\n Color mode used is not supported for deployment on MPU target \n Please use RGB format")
            if cfg.preprocessing.resizing.aspect_ratio != "fit":
                raise ValueError("\n Aspect ratio used is not supported for deployment on MPU target \n Please use 'fit' aspect ratio")
        parse_deployment_section(cfg.deployment,
                                 legal=legal,
                                 legal_hw=legal_hw)

    # MLFlow section parsing
    parse_mlflow_section(cfg.mlflow)

    # Check that all datasets have the required directory structure
    cds = cfg.dataset

    if not cds.class_names and cfg.operation_mode in ("deployment","chain_qd") and cfg.hardware_type == "MPU":
        cds.class_names = get_class_names_from_file(cfg)
        print("[INFO] : Found {} classes in label file {}".format(len(cds.class_names), cfg.deployment.label_file_path))

    return cfg
