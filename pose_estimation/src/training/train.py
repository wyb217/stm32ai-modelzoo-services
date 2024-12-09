# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
from functools import partial
import numpy as np
import tensorflow as tf
from typing import Tuple, List, Dict, Optional

from logs_utils import log_to_file, log_last_epoch_history
from models_utils import model_summary
from models_mgt import load_model_for_training
from loss import spe_loss
from metrics import single_pose_heatmaps_oks
from preprocess import apply_rescaling
from postprocess import heatmaps_spe_postprocess
from heatmaps_train_model import HMTrainingModel
from common_training import set_frozen_layers, set_dropout_rate, get_optimizer
from callbacks import get_callbacks
from evaluate import evaluate

def setup_hm_model(cfg, model, loss, metrics, model_input_shape, seed=None):

    data_augmentation_cfg = cfg.data_augmentation.config if cfg.data_augmentation else None
    scale = cfg.preprocessing.rescaling.scale
    offset = cfg.preprocessing.rescaling.offset
    pixels_range = (offset, scale * 255 + offset)

    # Create the custom model
    hm_model = HMTrainingModel(model,loss,metrics,data_augmentation_cfg,pixels_range,model_input_shape[:2])

    return hm_model

def train(cfg: DictConfig = None, train_ds: tf.data.Dataset = None,
          valid_ds: tf.data.Dataset = None, test_ds: Optional[tf.data.Dataset] = None) -> str:

    output_dir = Path(HydraConfig.get().runtime.output_dir)
    saved_models_dir = os.path.join(output_dir, cfg.general.saved_models_dir)
    tensorboard_log_dir = os.path.join(output_dir, cfg.general.logs_dir)
    metrics_dir = os.path.join(output_dir, cfg.general.logs_dir, "metrics")

    # Log dataset and model info
    log_to_file(output_dir, f"Dataset : {cfg.dataset.name}")
    log_to_file(cfg.output_dir, (f"Model type : {cfg.general.model_type}"))
    if cfg.general.model_path:
        log_to_file(cfg.output_dir ,(f"Model file : {cfg.general.model_path}"))
    elif cfg.training.resume_training_from:
        log_to_file(cfg.output_dir ,(f"Resuming training from : {cfg.training.resume_training_from}"))

    # Load the model to train
    model = load_model_for_training(cfg)
    model_input_shape = model.input.shape[1:]

    # Set frozen layers. By default, all layers are trainable.
    model.trainable = True
    if cfg.training.frozen_layers and cfg.training.frozen_layers != "None":
        set_frozen_layers(model, frozen_layers=cfg.training.frozen_layers)

    # Set rate on dropout layer if any
    if cfg.training.dropout:
        set_dropout_rate(model, dropout_rate=cfg.training.dropout)

    model_summary(model)

    # Save the base model
    base_model = tf.keras.models.clone_model(model)
    base_model_path = os.path.join(saved_models_dir, "base_model.h5")
    base_model.save(base_model_path)

    train_ds_re = apply_rescaling(dataset=train_ds, scale=cfg.preprocessing.rescaling.scale,
                               offset=cfg.preprocessing.rescaling.offset)

    valid_ds_re = apply_rescaling(dataset=valid_ds, scale=cfg.preprocessing.rescaling.scale,
                               offset=cfg.preprocessing.rescaling.offset)
    
    heatmaps_spe_loss = partial(spe_loss,output_type='heatmaps',loss_type='mse')
    
    def oks(a,b): return single_pose_heatmaps_oks(a,b)

    train_model = setup_hm_model(cfg,model,heatmaps_spe_loss,[oks],model_input_shape)

    train_model.compile(optimizer=get_optimizer(cfg.training.optimizer))

    callbacks = get_callbacks(
                    cfg=cfg.training.callbacks,
                    saved_models_dir=saved_models_dir,
                    log_dir=tensorboard_log_dir,
                    metrics_dir=metrics_dir)
    
    start_time = timer()
    train_model.fit(train_ds_re, validation_data=valid_ds_re, epochs=cfg.training.epochs, callbacks=callbacks)
    end_time = timer()

    # Log the last epoch history
    last_epoch = log_last_epoch_history(cfg, output_dir)
    
    # Calculate and log the runtime in the log file
    fit_run_time = int(end_time - start_time)
    average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1),2)
    print("Training runtime: " + str(timedelta(seconds=fit_run_time))) 
    log_to_file(cfg.output_dir, (f"Training runtime : {fit_run_time} s\n" + f"Average time per epoch : {average_time_per_epoch} s"))

    # Set weights and models paths
    base_model_path = os.path.join(saved_models_dir, "base_model.h5")
    best_weights_path = os.path.join(saved_models_dir, "best_weights.h5")
    best_model_path = os.path.join(saved_models_dir, "best_model.h5")
    last_weights_path = os.path.join(saved_models_dir, "last_weights.h5")
    last_model_path = os.path.join(saved_models_dir, "last_model.h5")
    
    print("[INFO] Saved trained models:")
    print("  best model:", best_model_path)
    print("  last model:", last_model_path)
    
    model = tf.keras.models.load_model(base_model_path, compile=False)
    model.load_weights(best_weights_path)
    model.save(best_model_path)
    model.load_weights(last_weights_path)
    model.save(last_model_path)

    if test_ds:
        evaluate(cfg=cfg, eval_ds=test_ds, model_path_to_evaluate=best_model_path, name_ds="test_set")
    else:
        evaluate(cfg=cfg, eval_ds=valid_ds, model_path_to_evaluate=best_model_path, name_ds="validation_set")

    return best_model_path
