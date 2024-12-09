# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import hydra
import argparse
import mlflow
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from utils import get_config, mlflow_ini, cloud_connect, benchmark, model_is_quantized, log_to_file
from trainers import train
from evaluators import evaluate
from quantization import quantize
from deployment import deploy

def _process_mode(cfg):
     
    mode = cfg.operation_mode

    # Logging the operation_mode in the output_dir/stm32ai_main.log file
    log_to_file(cfg.output_dir, f'operation_mode: {mode}')

    if mode == "training":
        onnx_model_path, _ = train(cfg)
        print("\n [INFO] Training complete\n"
              f"Trained model saved at {onnx_model_path}")

    elif mode == "evaluation":
        _, _, eval_logs_path = evaluate(cfg)
        print("\n [INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")

    elif mode == "quantization":
        quantized_model_path, quantized_static_model_path = quantize(cfg)
        print("\n [INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")
    
    elif mode == "benchmarking":
        model_path_to_benchmark = Path(cfg.model.onnx_path)
        benchmark(cfg, model_path_to_benchmark=model_path_to_benchmark)

    elif mode == "deployment":
        deploy(cfg)
    
    elif mode == "chain_tqe":
        float_onnx_model_path, best_float_onnx_model_path = train(cfg)
        print("[INFO] Training complete\n"
              f"Trained model saved at {float_onnx_model_path}")
        
        cfg.model.onnx_path = best_float_onnx_model_path
        print("[INFO] Evaluating float model")
        _, _, eval_logs_path = evaluate(cfg)
        print("[INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")
        
        quantized_model_path, quantized_static_model_path = quantize(cfg)
        print("[INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")
        
        cfg.model.onnx_path = quantized_model_path
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        _, _, eval_logs_path = evaluate(cfg)
        print("[INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")
        
    elif mode == "chain_eqe":
        if model_is_quantized(cfg.model.onnx_path):
            raise ValueError("Tried to run chain_eqe on a quantized ONNX model. \n"
                             "Chain_eqe can only be run on float ONNX models. \n"
                             "If you wish to evaluate a quantized ONNX model, use the 'evaluate' mode instead.")
        
        _, _, eval_logs_path = evaluate(cfg)
        print("[INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")
        
        quantized_model_path, quantized_static_model_path = quantize(cfg)
        print("[INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")
        
        cfg.model.onnx_path = quantized_model_path
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        _, _, eval_logs_path = evaluate(cfg)
        print("[INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")
        
    elif mode == "chain_tqeb":
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stedgeai.version)

        float_onnx_model_path, best_float_onnx_model_path = train(cfg)
        print("[INFO] Training complete\n"
              f"Trained model saved at {float_onnx_model_path}")
        
        cfg.model.onnx_path = best_float_onnx_model_path
        print("[INFO] Evaluating float model")
        _, _, eval_logs_path = evaluate(cfg)
        print("[INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")
        
        quantized_model_path, quantized_static_model_path = quantize(cfg)
        print("[INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")
        
        cfg.model.onnx_path = quantized_model_path
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        _, _, eval_logs_path = evaluate(cfg)
        print("[INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")
        
        benchmark(cfg, model_path_to_benchmark=quantized_static_model_path,
                  credentials=credentials)
        
    elif mode == "chain_eqeb":
        if model_is_quantized(cfg.model.onnx_path):
            raise ValueError("Tried to run chain_eqeb on a quantized ONNX model. \n"
                             "Chain_eqeb can only be run on float ONNX models. \n"
                             "If you wish to evaluate a quantized ONNX model, use the 'evaluate' mode instead.")
        
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stedgeai.version)

        _, _, eval_logs_path = evaluate(cfg)
        print("[INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")
        
        quantized_model_path, quantized_static_model_path = quantize(cfg)
        print("[INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")
        
        cfg.model.onnx_path = quantized_model_path
        p = Path(cfg.evaluation.logs_path)
        cfg.evaluation.logs_path = p.with_stem(p.stem + '_quantized')

        print("[INFO] Evaluating quantized model")
        _, _, eval_logs_path = evaluate(cfg)
        print("[INFO] Evaluation complete. \n"
              f"Evaluation logs saved at {eval_logs_path}")
        
        benchmark(cfg, model_path_to_benchmark=quantized_static_model_path,
                  credentials=credentials)
        
    elif mode == "chain_qb":
        if model_is_quantized(cfg.model.onnx_path):
            raise ValueError("Tried to run chain_qb on a quantized ONNX model. \n"
                             "Chain_qb can only be run on float ONNX models.")
        
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stedgeai.version)

        quantized_model_path, quantized_static_model_path = quantize(cfg)
        print("[INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")
        
        benchmark(cfg, model_path_to_benchmark=quantized_static_model_path,
                  credentials=credentials)
        
    elif mode == "chain_qd":
        if model_is_quantized(cfg.model.onnx_path):
            raise ValueError("Tried to run chain_qd on a quantized ONNX model. \n"
                             "Chain_qd can only be run on float ONNX models.")
        
        credentials = None
        if cfg.tools.stedgeai.on_cloud:
            _, _, credentials = cloud_connect(stm32ai_version=cfg.tools.stedgeai.version)

        quantized_model_path, quantized_static_model_path = quantize(cfg)
        print("[INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n"
              f"Quantized model with static input shape saved at {quantized_static_model_path}")
        
        deploy(cfg, model_path_to_deploy=quantized_static_model_path)
        
    else:
        raise ValueError(f"Invalid operation mode: {mode}")

    if mode in ['benchmarking', 'chain_tqeb', 'chain_qb', 'chain_eqeb']:
        mlflow.log_param("stm32ai_version", cfg.tools.stedgeai.version)
        mlflow.log_param("target", cfg.benchmarking.board)

    # logging the completion of the chain
    log_to_file(cfg.output_dir, f'Operation finished: {mode}')
        

@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.
 
    Args:
        cfg: Configuration dictionary.
 
    Returns:
        None
    """

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    mlflow_ini(cfg)

    _process_mode(cfg)

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