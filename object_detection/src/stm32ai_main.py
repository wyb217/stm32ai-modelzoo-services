# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022-2023 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
import os
import sys
import hydra
import argparse
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
import mlflow
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), '../../common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/benchmarking'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/deployment'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/quantization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/optimization'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/data_augmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/training'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../common/onnx_utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../deployment'))
sys.path.append(os.path.join(os.path.dirname(__file__), './data_augmentation'))
sys.path.append(os.path.join(os.path.dirname(__file__), './models'))
sys.path.append(os.path.join(os.path.dirname(__file__), './preprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), './postprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), './training'))
sys.path.append(os.path.join(os.path.dirname(__file__), './utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), './evaluation'))
sys.path.append(os.path.join(os.path.dirname(__file__), './quantization'))
sys.path.append(os.path.join(os.path.dirname(__file__), './prediction'))


from logs_utils import mlflow_ini, log_to_file
from gpu_utils import set_gpu_memory_limit
from cfg_utils import get_random_seed
from parse_config import get_config
from train import train
from evaluate import evaluate
from quantize import quantize
from predict import predict
from common_benchmark import benchmark, cloud_connect
from deploy import deploy, deploy_mpu


# This function turns Tensorflow's eager mode on and off.
# Eager mode is for debugging the Model Zoo code and is slower.
# Do not set argument to True to avoid runtime penalties.
tf.config.run_functions_eagerly(False)


def process_mode(cfg: DictConfig):
    """
    Execution of the various services

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """
    mode = cfg.operation_mode

    mlflow.log_param("model_path", cfg.general.model_path)
    # logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(cfg.output_dir, f'operation_mode: {mode}')

    if mode == "training":
        train(cfg)
        print("[INFO] training complete")

    elif mode == "evaluation":
        evaluate(cfg)
        print("[INFO] evaluation complete")

    elif mode == "quantization":
        quantize(cfg)
        print("[INFO] quantization complete")

    elif mode == "prediction":
        predict(cfg)
        print("[INFO] prediction complete")

    elif mode == 'benchmarking':
        benchmark(cfg)
        print("[INFO] benchmarking complete")

    elif mode == 'deployment':
        if cfg.hardware_type == "MPU":
            deploy_mpu(cfg)
        else:
            deploy(cfg)
        print("[INFO] deployment complete")
        print('[INFO] : Please on STM32N6570-DK toggle the boot switches to the left and power cycle the board.')

    elif mode == 'chain_tqe':
        trained_model_path = train(cfg)
        quantized_model_path = quantize(cfg, model_path=trained_model_path)
        evaluate(cfg, model_path=quantized_model_path)
        print("Trained model path:", trained_model_path)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_tqe complete")

    elif mode == 'chain_tqeb':
        credentials = None
        if cfg.tools.stm32ai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)
        trained_model_path = train(cfg)
        quantized_model_path = quantize(cfg, model_path=trained_model_path)
        evaluate(cfg, model_path=quantized_model_path)
        benchmark(cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
        print("Trained model path:", trained_model_path)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_tqeb complete")

    elif mode == 'chain_eqe':
        evaluate(cfg)
        quantized_model_path = quantize(cfg)
        evaluate(cfg, model_path=quantized_model_path)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_eqe complete")

    elif mode == 'chain_eqeb':
        credentials = None
        if cfg.tools.stm32ai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)
        evaluate(cfg)
        quantized_model_path = quantize(cfg)
        evaluate(cfg, model_path=quantized_model_path)
        benchmark(cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_eqeb complete")

    elif mode == 'chain_qb':
        credentials = None
        if cfg.tools.stm32ai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stm32ai.version)
        quantized_model_path = quantize(cfg)
        benchmark(cfg, model_path_to_benchmark=quantized_model_path, credentials=credentials)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_qb complete")

    elif mode == 'chain_qd':
        quantized_model_path = quantize(cfg)
        if cfg.hardware_type == "MPU":
            deploy_mpu(cfg, model_path_to_deploy=quantized_model_path)
        else:
            deploy(cfg, model_path_to_deploy=quantized_model_path)
        print("Quantized model path:", quantized_model_path)
        print("[INFO] chain_qd complete")

    elif mode == 'prediction':
        predict(cfg)

    else:
        raise RuntimeError(f"Internal error: invalid operation mode: {mode}")

    if mode in ['benchmarking', 'chain_tbqeb', 'chain_qb', 'chain_eqeb']:
        mlflow.log_param("stm32ai_version", cfg.tools.stm32ai.version)
        mlflow.log_param("target", cfg.benchmarking.board)

    # logging the completion of the chain
    log_to_file(cfg.output_dir, f'operation finished: {mode}')


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
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    mlflow_ini(cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)

    # The default hardware type is "MCU".
    process_mode(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, default='', help='Path to folder containing configuration file')
    parser.add_argument('--config-name', type=str, default='user_config', help='name of the configuration file')

    # Add arguments to the parser
    parser.add_argument('params', nargs='*',
                        help='List of parameters to over-ride in config.yaml')
    args = parser.parse_args()

    # Call the main function
    main()

    # log the config_path and config_name parameters
    mlflow.log_param('config_path', args.config_path)
    mlflow.log_param('config_name', args.config_name)
    mlflow.end_run()
