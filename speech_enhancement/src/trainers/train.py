# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import torch
import copy
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from trainers import MagSpecTrainer
from dataset_utils import load_dataset_from_cfg
from utils import plot_training_metrics, log_to_file
import pandas as pd
import models
import preprocessing


def _train(model,
           n_epochs,
           train_dl,
           valid_dl,
           optimizer,
           loss,
           batching_strat,
           device,
           device_memory_fraction,
           preproc_args,
           regularization_args,
           ckpt_path,
           logs_path,
           snapshot_path,
           save_every,
           early_stopping,
           early_stopping_patience,
           reference_metric
           ):
    
    trainer = MagSpecTrainer(model=model,
                             optimizer=optimizer,
                             train_data=train_dl,
                             valid_data=valid_dl,
                             loss=loss,
                             batching_strat=batching_strat,
                             device=device,
                             device_memory_fraction=device_memory_fraction,
                             save_every=save_every,
                             ckpt_path=ckpt_path,
                             logs_path=logs_path,
                             snapshot_path=snapshot_path,
                             early_stopping=early_stopping,
                             early_stopping_patience=early_stopping_patience,
                             reference_metric=reference_metric,
                             **preproc_args,
                             **regularization_args
                             )
    model, best_model = trainer.train(n_epochs=n_epochs)

    return model, best_model

def train(cfg):
    # Logging to stm32aiout file
    log_to_file(cfg.output_dir, f"Dataset: {cfg.dataset.name}")

    # Initialize preproc pipelines 
    loss = cfg.training.loss

    pipeline_args = copy.copy(cfg.preprocessing)

    del pipeline_args["pipeline_type"]

    input_pipeline = getattr(preprocessing, cfg.preprocessing.pipeline_type)(
        magnitude=False, **pipeline_args)

    if loss == "spec_mse":
        # If using 
        train_target_pipeline = input_pipeline
    elif loss in ['wave_mse', 'wave_sisnr', 'wave_snr']:
        train_target_pipeline = preprocessing.IdentityPipeline(peak_normalize=pipeline_args["peak_normalize"])
    else:
        raise ValueError("Invalid loss type. Should be one of 'spec_mse', 'wave_mse',"
                         f"'wave_sisnr', 'wave_snr', but was {loss}")

    valid_target_pipeline = preprocessing.IdentityPipeline(peak_normalize=pipeline_args["peak_normalize"])

    # Load training dataset
    train_ds = load_dataset_from_cfg(cfg,
                                     set="train",
                                     n_clips=cfg.dataset.num_training_samples,
                                     val_split=cfg.dataset.num_validation_samples,
                                     input_pipeline=input_pipeline,
                                     target_pipeline=train_target_pipeline)
    
    train_dl = DataLoader(train_ds,
                          batch_size=cfg.training.batch_size,
                          num_workers=cfg.training.num_dataloader_workers,
                          shuffle=cfg.training.shuffle)

    # Load validation dataset
    valid_ds = load_dataset_from_cfg(cfg,
                                     set="valid",
                                     n_clips=None,
                                     val_split=cfg.dataset.num_validation_samples,
                                     input_pipeline=input_pipeline,
                                     target_pipeline=valid_target_pipeline)
    
    # Here, batch size is forced to 1 to avoid padding/trimming during validation
    valid_dl = DataLoader(valid_ds, batch_size=1)

    # Load model
    model_type = cfg.model.model_type
    model_specific_args = cfg.model_specific
    model = getattr(models, model_type)(**model_specific_args)

    log_to_file(cfg.output_dir, f"Model type: {cfg.model.model_type}")
    
    # If a state dict is given in cfg, load it
    if cfg.model.state_dict_path:
        state_dict = torch.load(cfg.model.state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded state dict at {cfg.model.state_dict_path}")
        log_to_file(cfg.output_dir, f"Loaded model state dict at: {cfg.model.state_dict_path}")
    
    # Initialize optimizer
    optimizer = getattr(torch.optim, cfg.training.optimizer)(
        params=model.parameters(), **cfg.training.optimizer_arguments)
    
    # Gather preprocessing args that need to be passed to trainer
    preproc_args = {"sampling_rate":cfg.preprocessing.sample_rate,
                    "frame_length":cfg.preprocessing.win_length,
                    "hop_length":cfg.preprocessing.hop_length,
                    "n_fft":cfg.preprocessing.n_fft,
                    "center":cfg.preprocessing.center}

    ckpt_path = Path(cfg.output_dir, cfg.training.ckpt_path)
    logs_path = Path(cfg.output_dir, 'training_logs', cfg.training.logs_filename)
    
    # If user provides a training snapshot, use it.
    if cfg.training.snapshot_path is None:
        snapshot_path = Path(cfg.output_dir, 'training_logs', 'training_snapshot.pth')
    else:
        snapshot_path = cfg.training.snapshot_path
    
    # If snapshot file exists, it will be loaded automatically by the trainer
    # Log this to file
    if snapshot_path.exists():
        log_to_file(cfg.output_dir, f"Loaded training snapshot at {snapshot_path}")

    # Make dirs if necessary
    logs_path.parent.mkdir(parents=False, exist_ok=True)
    ckpt_path.mkdir(parents=False, exist_ok=True)

    if cfg.training.regularization is None:
        cfg.training.regularization = {}
    model, best_model = _train(model=model,
                                n_epochs=cfg.training.epochs,
                                train_dl=train_dl,
                                valid_dl=valid_dl,
                                optimizer=optimizer,
                                loss=loss,
                                batching_strat=cfg.training.batching_strategy,
                                device=cfg.training.device,
                                device_memory_fraction=cfg.general.gpu_memory_limit,
                                preproc_args=preproc_args,
                                regularization_args=cfg.training.regularization,
                                ckpt_path=ckpt_path,
                                logs_path=logs_path,
                                snapshot_path=snapshot_path,
                                save_every=cfg.training.save_every,
                                early_stopping=cfg.training.early_stopping,
                                early_stopping_patience=cfg.training.early_stopping_patience,
                                reference_metric=cfg.training.reference_metric
                                )
    
    # Export to ONNX
    # Here we assume the model's input shape is (batch, n_fft // 2 + 1, sequence_length)
    # Might change this later and expose the input shape in cfg
    # We also assume it only has one input & output
    # NOTE : Change this when adding support for decomposed LSTM
    
    model.eval()
    dummy_tensor = torch.ones((1, cfg.preprocessing.n_fft // 2 + 1, 10))
    model.to("cpu")
    onnx_model_path = Path(cfg.output_dir, cfg.general.saved_models_dir, 'trained_model.onnx')
    onnx_model_path.parent.mkdir(exist_ok=True)
    torch.onnx.export(model,
                    dummy_tensor,
                    onnx_model_path,
                    export_params=True,
                    opset_version=cfg.training.opset_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input":{2:"seq_len"},
                                    "output":{2:"seq_len"}} # Dynamic sequence length axes
                    )
    
    # Same with best model 
    
    best_model.eval()
    best_model.to("cpu")
    best_onnx_model_path = Path(cfg.output_dir, cfg.general.saved_models_dir, 'best_trained_model.onnx')
    torch.onnx.export(best_model,
                    dummy_tensor,
                    best_onnx_model_path,
                    export_params=True,
                    opset_version=cfg.training.opset_version,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input":{2:"seq_len"},
                                    "output":{2:"seq_len"}} # Dynamic sequence length axes
                    )

    # Plot training & validation metrics

    metrics_df = pd.read_csv(logs_path)
    fig = plot_training_metrics(metrics_df=metrics_df, figsize=(12, 15))

    if cfg.general.display_figures:
        plt.show()
    plt.savefig(Path(logs_path.parent, "training_metrics.png"))

    return onnx_model_path, best_onnx_model_path






    

    
