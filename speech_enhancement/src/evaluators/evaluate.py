# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import copy
from pathlib import Path
from torch.utils.data import DataLoader
from dataset_utils import load_dataset_from_cfg
from evaluators import MagSpecONNXEvaluator, MagSpecTorchEvaluator
from utils import model_is_quantized, log_to_file
import models
import preprocessing
import mlflow

def _evaluate(eval_dl,
              preproc_args,
              logs_path,
              device="cpu",
              device_memory_fraction=0.5,
              fixed_sequence_length=None,
              model=None,
              model_checkpoint=None,
              onnx_model_path=None):
    
    if onnx_model_path is not None:
        evaluator = MagSpecONNXEvaluator(model_path=onnx_model_path,
                                         eval_data=eval_dl,
                                         logs_path=logs_path,
                                         fixed_sequence_length=fixed_sequence_length,
                                         **preproc_args)
    elif model is not None and model_checkpoint is not None:
        evaluator = MagSpecTorchEvaluator(model=model,
                                          model_checkpoint=model_checkpoint,
                                          eval_data=eval_dl,
                                          logs_path=logs_path,
                                          device=device,
                                          device_memory_fraction=device_memory_fraction,
                                          **preproc_args
                                          )
    else:
        raise ValueError("Must provide either an ONNX model or a torch model class and a state_dict")
    
    metrics_dict, metrics_array = evaluator.evaluate()
    
    return metrics_dict, metrics_array

def evaluate(cfg):

    pipeline_args = copy.copy(cfg.preprocessing)

    del pipeline_args["pipeline_type"]

    input_pipeline = getattr(preprocessing, cfg.preprocessing.pipeline_type)(
        magnitude=False, **pipeline_args)

    target_pipeline = preprocessing.IdentityPipeline(peak_normalize=pipeline_args["peak_normalize"])


    # Load evaluation dataset
    eval_ds = load_dataset_from_cfg(cfg,
                                    set="test",
                                    n_clips=cfg.dataset.num_test_samples,
                                    input_pipeline=input_pipeline,
                                    target_pipeline=target_pipeline)
    
    eval_dl = DataLoader(eval_ds,
                         batch_size=1)

    # Load model
    # If onnx model is provided in config, use it 
    # Else, look for model type and model state dict.
    # If none are provided, raise error.
    onnx_model_path = None
    model = None
    model_checkpoint = None

    if cfg.model.onnx_path:
        onnx_model_path = cfg.model.onnx_path
        # Some logging
        model_type = "Quantized" if model_is_quantized(onnx_model_path) else "Float"
    elif cfg.model.model_type and cfg.model.state_dict_path:
        model_type = cfg.model.model_type
        model_specific_args = cfg.model_specific
        model = getattr(models, model_type)(**model_specific_args)
        model_checkpoint = cfg.model.state_dict_path
        model_type = "Float"
    else:
        raise ValueError("Must provide either 'cfg.model.onnx_path' \n"
                         " or both 'cfg.model_type and cfg.model.state_dict_path")
    
    log_to_file(cfg.output_dir, f"{model_type} model evaluation on dataset {cfg.dataset.name}")
    


    # Gather preprocessing args that need to be passed to evaluator
    preproc_args = {"sampling_rate":cfg.preprocessing.sample_rate,
                    "frame_length":cfg.preprocessing.win_length,
                    "hop_length":cfg.preprocessing.hop_length,
                    "n_fft":cfg.preprocessing.n_fft,
                    "center":cfg.preprocessing.center}
    if cfg.evaluation.logs_path is None:
        logs_path = Path(cfg.output_dir, "eval_logs/")
    else:
        logs_path = Path(cfg.output_dir, cfg.evaluation.logs_path) 

    # Make dirs if necessary
    logs_path.mkdir(parents=False, exist_ok=True)

    metrics_dict, metrics_array = _evaluate(eval_dl=eval_dl,
                                            preproc_args=preproc_args,
                                            logs_path=logs_path,
                                            device=cfg.evaluation.device,
                                            device_memory_fraction=cfg.general.gpu_memory_limit,
                                            fixed_sequence_length=cfg.evaluation.fixed_sequence_lenth,
                                            model=model,
                                            model_checkpoint=model_checkpoint,
                                            onnx_model_path=onnx_model_path
                                            )
    print("[INFO] Average metrics on test set : ")
    for key in metrics_dict.keys():
        print(f"{key} : {metrics_dict[key]}")
        log_to_file(cfg.output_dir, f"{model_type} {key} : {metrics_dict[key]}")
    # Log in mlflow
    mlflow.log_metrics(metrics_dict)
        
    return metrics_dict, metrics_array, logs_path