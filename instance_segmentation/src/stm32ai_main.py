# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
from hydra.core.hydra_config import HydraConfig
import hydra

import tensorflow as tf
from omegaconf import DictConfig
import mlflow
import argparse
from typing import Optional


sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/benchmarking'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/deployment'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/quantization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/optimization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../deployment'))
sys.path.append(os.path.join(os.path.dirname(__file__), './preprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), './postprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), './training'))
sys.path.append(os.path.join(os.path.dirname(__file__), './utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), './evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), './quantization'))
sys.path.append(os.path.join(os.path.dirname(__file__), './prediction'))
sys.path.append(os.path.join(os.path.dirname(__file__), './models'))




from logs_utils import mlflow_ini
from gpu_utils import set_gpu_memory_limit
from cfg_utils import get_random_seed
from parse_config import get_config
#from deploy import deploy
from predict import predict
from common_benchmark import benchmark
from common_benchmark import cloud_connect
#from deploy import deploy, deploy_mpu
from logs_utils import log_to_file

from deploy import deploy


def process_mode(mode: str = None,
                 configs: DictConfig = None,) -> None:

    mlflow.log_param("model_path", configs.general.model_path)
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(configs.output_dir, f'operation_mode: {mode}')
    # Check the selected mode and perform the corresponding operation
    if mode == 'deployment':
        if configs.hardware_type == "MPU":
            raise ValueError(f"Invalid hardware. Only STM32N6570-DK boards are supported for deployment.")
        else:
            deploy(cfg=configs)
        print('[INFO] : Deployment complete.')
        print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')
    elif mode == 'prediction':
        predict(cfg=configs)
        print('[INFO] : Prediction complete.')
    elif mode == 'benchmarking':
        benchmark(cfg=configs, custom_objects=None)
        print('[INFO] : Benchmark complete.')
    # Raise an error if an invalid mode is selected
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Record the whole hydra working directory to get all info
    mlflow.log_artifact(configs.output_dir)
    if mode =='benchmarking':
        mlflow.log_param("stm32ai_version", configs.tools.stm32ai.version)
        mlflow.log_param("target", configs.benchmarking.board)
    # logging the completion of the chain
    log_to_file(configs.output_dir, f'operation finished: {mode}')


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """

    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
            print(f"[INFO] Setting upper limit of usable GPU memory to {int(cfg.general.gpu_memory_limit)}GBytes.")
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().run.dir
    mlflow_ini(cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # Process the selected mode
    process_mode(mode=cfg.operation_mode, configs=cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')
    # add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
