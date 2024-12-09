# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
from munch import DefaultMunch

from cfg_utils import aspect_ratio_dict, postprocess_config_dict, check_config_attributes, \
                      parse_tools_section, parse_benchmarking_section, parse_mlflow_section, \
                      parse_general_section, parse_top_level, parse_prediction_section, \
                      parse_deployment_section, check_hardware_type, get_class_names_from_file




def parse_dataset_section(cfg: DictConfig, mode: str = None, mode_groups: DictConfig = None, hardware_type: str = None) -> None:


    legal = ["name", "class_names", "classes_file_path", "training_path", "validation_path", "validation_split", "test_path",
             "quantization_path", "quantization_split", "check_image_files", "seed"]

    required = []
    one_or_more = []
    if mode in ["deploymenet", "prediction"]:
        one_or_more += ["class_names", "classes_file_path"]
    check_config_attributes(cfg, specs={"legal": legal, "all": required, "one_or_more": one_or_more},
                            section="dataset")
    
def parse_preprocessing_section(cfg: DictConfig,
                                mode: str = None) -> None:
    """
            This function checks the preprocessing section of the config file.

            Arguments:
                cfg (DictConfig): The entire configuration file as a DefaultMunch dictionary.
                mode (str): the operation mode for example: 'quantization', 'evaluation'...

            Returns:
                None
        """

    legal = ["rescaling", "resizing", "color_mode"]
    if mode == 'deployment':
        # removing the obligation to have rescaling for the 'deployment' mode
        required = ["resizing", "color_mode"]
        check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="preprocessing")
    else:
        required = legal
        check_config_attributes(cfg, specs={"legal": legal, "all": required}, section="preprocessing")
        legal = ["scale", "offset"]
        check_config_attributes(cfg.rescaling, specs={"legal": legal, "all": legal}, section="preprocessing.rescaling")

    legal = ["interpolation", "aspect_ratio"]
    if cfg.resizing.aspect_ratio == "fit":
        required = ["interpolation", "aspect_ratio"]
    else:
        required = ["aspect_ratio"]

    check_config_attributes(cfg.resizing, specs={"legal": legal, "all": required}, section="preprocessing.resizing")

    # Check the of aspect ratio value
    aspect_ratio = cfg.resizing.aspect_ratio
    if aspect_ratio not in aspect_ratio_dict:
        raise ValueError(f"\nUnknown or unsupported value for `aspect_ratio` attribute. Received {aspect_ratio}\n"
                         f"Supported values: {list(aspect_ratio_dict.keys())}.\n"
                         "Please check the 'preprocessing.resizing' section of your configuration file.")

    if aspect_ratio == "fit":
        # Check resizing interpolation value
        check_config_attributes(cfg.resizing, specs={"all": ["interpolation"]}, section="preprocessing.resizing")
        interpolation_methods = ["bilinear", "nearest", "area", "lanczos3", "lanczos5", "bicubic", "gaussian",
                                 "mitchellcubic"]
        if cfg.resizing.interpolation not in interpolation_methods:
            raise ValueError(f"\nUnknown value for `interpolation` attribute. Received {cfg.resizing.interpolation}\n"
                             f"Supported values: {interpolation_methods}\n"
                             "Please check the 'preprocessing.resizing' section of your configuration file.")

    # Check color mode value
    color_modes = ["grayscale", "rgb", "rgba"]
    if cfg.color_mode not in color_modes:
        raise ValueError(f"\nUnknown value for `color_mode` attribute. Received {cfg.color_mode}\n"
                         f"Supported values: {color_modes}\n"
                         "Please check the 'preprocessing' section of your configuration file.")
    

def get_config(config_data: DictConfig) -> DefaultMunch:

    config_dict = OmegaConf.to_container(config_data)

    # Restore booleans, numerical expressions and tuples
    # Expand environment variables
    postprocess_config_dict(config_dict)
    
    # Top level section parsing
    cfg = DefaultMunch.fromDict(config_dict)

    mode_groups = DefaultMunch.fromDict({
    "training": ["training", "chain_tbqeb", "chain_tqe"],
    "evaluation": ["evaluation", "chain_tbqeb", "chain_tqe", "chain_eqe", "chain_eqeb"],
    "quantization": ["quantization", "chain_tbqeb", "chain_tqe", "chain_eqe",
                        "chain_qb", "chain_eqeb", "chain_qd"],
    "benchmarking": ["benchmarking", "chain_tbqeb", "chain_qb", "chain_eqeb"],
    "deployment": ["deployment", "chain_qd"]})

    mode_choices = ["prediction", "deployment","benchmarking"]
    legal = ["general", "operation_mode","training", "dataset", "preprocessing", "prediction", "postprocessing",
             "tools", "benchmarking", "deployment", "mlflow", "hydra"]
    parse_top_level(cfg,
                    mode_groups=mode_groups,
                    mode_choices=mode_choices,
                    legal=legal)
    print(f"[INFO] : Running `{cfg.operation_mode}` operation mode")

    # General section parsing
    if not cfg.general:
        cfg.general = DefaultMunch.fromDict({"project_name": "<unnamed>"})
    legal = ["project_name", "model_path", "logs_dir", "display_figures", "global_seed", 
             "gpu_memory_limit", "model_type", "num_threads_tflite"]
    required = []
    parse_general_section(cfg.general,
                          mode=cfg.operation_mode,
                          mode_groups=mode_groups,
                          legal=legal,
                          required=required,
                          output_dir = HydraConfig.get().runtime.output_dir)

    # Select hardware_type from yaml information
    check_hardware_type(cfg,
                        mode_groups)

    # Dataset section parsing
    if not cfg.dataset:
        cfg.dataset = DefaultMunch.fromDict({})
    parse_dataset_section(cfg.dataset,
                          mode=cfg.operation_mode,
                          mode_groups=mode_groups,
                          hardware_type=cfg.hardware_type)

    # Preprocessing section parsing
    if cfg.operation_mode in ["prediction", "deploymenet"]:
        parse_preprocessing_section(cfg.preprocessing,
                                    mode=cfg.operation_mode)

    # Prediction section parsing
    if cfg.operation_mode == "prediction":
        parse_prediction_section(cfg.prediction)

    # Tools section parsing
    if cfg.operation_mode in (mode_groups.benchmarking + mode_groups.deployment):
        parse_tools_section(cfg.tools,
                            cfg.operation_mode,
                            cfg.hardware_type)
        
    # Benchmarking section parsing
    if cfg.operation_mode in mode_groups.benchmarking:
        parse_benchmarking_section(cfg.benchmarking)
        if cfg.hardware_type == "MPU":
            if not (cfg.tools.stm32ai.on_cloud):
                print("Target selected for benchmark :", cfg.benchmarking.board)
                print("Offline benchmarking for MPU is not yet available please use online benchmarking")
                exit(1)

    # Deployment section parsing
    if cfg.operation_mode in mode_groups.deployment:
        if cfg.hardware_type == "MCU":
            legal = ["c_project_path", "IDE", "verbosity", "hardware_setup"]
            legal_hw = ["serie", "board", "stlink_serial_number"]
        else:
            legal = ["c_project_path", "board_deploy_path", "verbosity", "hardware_setup"]
            legal_hw = ["serie", "board", "ip_address"]
            if cfg.preprocessing.color_mode != "rgb":
                raise ValueError("\n Color mode used is not supported for deployment on MPU target \n Please use RGB format")
            if cfg.preprocessing.resizing.aspect_ratio != "fit":
                raise ValueError("\n Aspect ratio used is not supported for deployment on MPU target \n Please use FIT aspect ratio")
        parse_deployment_section(cfg.deployment,
                                 legal=legal,
                                 legal_hw=legal_hw)

    # MLFlow section parsing
    parse_mlflow_section(cfg.mlflow)

    # Check that all datasets have the required directory structure
    cds = cfg.dataset
    if not cds.class_names and cfg.operation_mode not in ("benchmarking"):
        cds.class_names = get_class_names_from_file(cfg)
        print("[INFO] : Found {} classes in label file {}".format(len(cds.class_names), cfg.dataset.classes_file_path))

    return cfg
