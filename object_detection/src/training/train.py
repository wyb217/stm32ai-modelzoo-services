# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
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
import numpy as np
import tensorflow as tf

# Suppress Tensorflow warnings
import logging
logging.getLogger('mlflow.tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from cfg_utils import parse_random_periodic_resizing
from logs_utils import log_to_file, log_last_epoch_history
from models_utils import model_summary
from models_mgt import model_family, load_model_for_training
from tiny_yolo_v2 import tiny_yolo_v2
from anchor_boxes_utils import get_sizes_ratios_ssd_v1, get_sizes_ratios_ssd_v2, get_fmap_sizes, get_anchor_boxes
from datasets import get_training_data_loaders
from ssd_train_model import SSDTrainingModel
from yolo_train_model import YoloTrainingModel
from yolo_x_train_model import YoloXTrainingModel
from common_training import set_frozen_layers, set_dropout_rate, get_optimizer
from callbacks import get_callbacks
from evaluate import evaluate


def set_up_ssd_model(cfg, model, input_shape=None, num_labels=None, 
                     val_dataset_size=None, pixels_range=None):

    # Get the anchor boxes
    fmap_sizes = get_fmap_sizes(cfg.general.model_type, input_shape)
    
    if cfg.general.model_type == "st_ssd_mobilenet_v1":
        anchor_sizes, anchor_ratios = get_sizes_ratios_ssd_v1(input_shape)
    elif cfg.general.model_type == "ssd_mobilenet_v2_fpnlite":
        anchor_sizes, anchor_ratios = get_sizes_ratios_ssd_v2(input_shape)
    
    anchor_boxes = get_anchor_boxes(
                        fmap_sizes,
                        input_shape[:2],
                        sizes=anchor_sizes,
                        ratios=anchor_ratios,
                        normalize=True,
                        clip_boxes=False)

    # Concatenate scores, boxes and anchors
    # to get a model suitable for training
    tmoutput = tf.keras.layers.Concatenate(axis=2, name='predictions')(model.outputs)
    train_model = tf.keras.models.Model(inputs=model.input, outputs=tmoutput)
    
    data_augmentation_cfg = cfg.data_augmentation.config if cfg.data_augmentation else None      
    num_anchors = np.shape(anchor_boxes)[0]
    cpp = cfg.postprocessing
    ssd_model = SSDTrainingModel(
                        train_model,
                        num_classes=len(cfg.dataset.class_names),
                        num_anchors=num_anchors,
                        num_labels=num_labels,
                        val_dataset_size=val_dataset_size,
                        anchor_boxes=anchor_boxes,
                        data_augmentation_cfg=data_augmentation_cfg,
                        pixels_range=pixels_range,
                        pos_iou_threshold=0.5,
                        neg_iou_threshold=0.3,
                        max_detection_boxes=cpp.max_detection_boxes,
                        nms_score_threshold=cpp.confidence_thresh,
                        nms_iou_threshold=cpp.NMS_thresh,
                        metrics_iou_threshold=cpp.IoU_eval_thresh)

    return ssd_model


def set_up_yolo_model(cfg, model, input_shape=None, num_labels=None, val_dataset_size=None, pixels_range=None):

    cpp = cfg.postprocessing
    
    # If multi-resolution is used, we need to check that the
    # random image sizes are compatible with the network stride.
    if cfg.data_augmentation:
        cda = cfg.data_augmentation.config
        message = "\nPlease check the `random_periodic_resizing` section in your configuration file."

        if "random_periodic_resizing" in cda:
            # Parse the random image sizes and check that 
            # they are compatible with the network stride
            image_sizes = parse_random_periodic_resizing(cda.random_periodic_resizing, cpp.network_stride)
            cda.random_periodic_resizing.image_sizes = image_sizes

    print("Using Yolo anchors:")
    for anchor in cpp.yolo_anchors:
        print(" ", anchor)

    data_augmentation_cfg = cfg.data_augmentation.config if cfg.data_augmentation else None      

    # Create the custom model
    yolo_model = YoloTrainingModel(
                        model,
                        network_stride=cpp.network_stride,
                        num_classes=len(cfg.dataset.class_names),
                        num_labels=num_labels,
                        anchors=cpp.yolo_anchors,
                        data_augmentation_cfg=data_augmentation_cfg,
                        val_dataset_size=val_dataset_size,
                        pixels_range=pixels_range,
                        image_size=input_shape[:2],
                        max_detection_boxes=cpp.max_detection_boxes,
                        nms_score_threshold=cpp.confidence_thresh,
                        nms_iou_threshold=cpp.NMS_thresh,
                        metrics_iou_threshold=cpp.IoU_eval_thresh)
    return yolo_model


def set_up_yolo_x_model(cfg, model, input_shape=None, num_labels=None, val_dataset_size=None, pixels_range=None):

    cpp = cfg.postprocessing
    ctm = cfg.training.model

    print("Using Yolo anchors:")
    for anchor in cpp.yolo_anchors:
        print(" ", anchor)
    print("Using depth_mul: ",ctm.depth_mul)
    print("Using width_mul: ",ctm.width_mul)

    data_augmentation_cfg = cfg.data_augmentation.config if cfg.data_augmentation else None      

    # Create the custom model
    yolo_model = YoloXTrainingModel(
                        model,
                        network_stride=cpp.network_stride,
                        num_classes=len(cfg.dataset.class_names),
                        num_labels=num_labels,
                        anchors=cpp.yolo_anchors,
                        data_augmentation_cfg=data_augmentation_cfg,
                        val_dataset_size=val_dataset_size,
                        pixels_range=pixels_range,
                        image_size=input_shape[:2],
                        max_detection_boxes=cpp.max_detection_boxes,
                        nms_score_threshold=cpp.confidence_thresh,
                        nms_iou_threshold=cpp.NMS_thresh,
                        metrics_iou_threshold=cpp.IoU_eval_thresh)
    return yolo_model


def train(cfg: DictConfig):

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

    # Load the model to train (Model Zoo model,
    # user model or training resume model)
    model = load_model_for_training(cfg)    
    model_input_shape = model.input.shape[1:]

    # Create and save the base model (compile it
    # to avoid warnings when reloading it)
    base_model = tf.keras.models.clone_model(model)
    base_model.compile()
    base_model_path = os.path.join(saved_models_dir, "base_model.h5")
    base_model.save(base_model_path)
    
    # Set frozen layers. By default, all layers are trainable.
    model.trainable = True
    if cfg.training.frozen_layers and cfg.training.frozen_layers != "None":
        set_frozen_layers(model, frozen_layers=cfg.training.frozen_layers)

    # Set rate on dropout layer if any
    if cfg.training.dropout:
        set_dropout_rate(model, dropout_rate=cfg.training.dropout)

    model_summary(model)

    # If multi-resolution data augmentation is used, the model
    # input shape must be replaced with (None, None, channels).
    if cfg.data_augmentation:
        if cfg.data_augmentation.config.random_periodic_resizing:
            channels = model_input_shape[-1]
            input_layer = tf.keras.Input(shape=(None, None, channels))
            x = model.layers[1](input_layer)
            for layer in model.layers[2:]:
                x = layer(x)
            model = tf.keras.Model(inputs=input_layer, outputs=x)

    # Create the data loaders
    train_ds, val_ds = get_training_data_loaders(cfg, image_size=model_input_shape[:2])
    
    print("Metrics calculation parameters:")
    print("  confidence threshold:", cfg.postprocessing.confidence_thresh)
    print("  NMS IoU threshold:", cfg.postprocessing.NMS_thresh)
    print("  max detection boxes:", cfg.postprocessing.max_detection_boxes)
    print("  metrics IoU threshold:", cfg.postprocessing.IoU_eval_thresh)

    scale = cfg.preprocessing.rescaling.scale
    offset = cfg.preprocessing.rescaling.offset
    pixels_range = (offset, scale * 255 + offset)

    # Get the number of groundtruth labels used in the datasets
    _, labels = iter(train_ds).next()
    num_labels = int(tf.shape(labels)[1])

    # Get the size of the validation set
    val_dataset_size = sum([x.shape[0] for x, _ in val_ds])

    if model_family(cfg.general.model_type) == "ssd":
        train_model = set_up_ssd_model(
                            cfg,
                            model,
                            input_shape=model_input_shape, 
                            num_labels=num_labels,
                            val_dataset_size=val_dataset_size,
                            pixels_range=pixels_range)
                                       
    elif model_family(cfg.general.model_type) == "yolo":
        train_model = set_up_yolo_model(
                            cfg,
                            model,
                            input_shape=model_input_shape,
                            num_labels=num_labels,
                            val_dataset_size=val_dataset_size,
                            pixels_range=pixels_range)
        
    elif model_family(cfg.general.model_type) == "st_yolo_x":
        train_model = set_up_yolo_x_model(
                            cfg,
                            model,
                            input_shape=model_input_shape,
                            num_labels=num_labels,
                            val_dataset_size=val_dataset_size,
                            pixels_range=pixels_range)

    train_model.compile(optimizer=get_optimizer(cfg.training.optimizer))    

    # Set up callbacks
    callbacks = get_callbacks(
                    cfg=cfg.training.callbacks,
                    num_classes=len(cfg.dataset.class_names),
                    iou_eval_threshold=cfg.postprocessing.IoU_eval_thresh,
                    saved_models_dir=saved_models_dir,
                    log_dir=tensorboard_log_dir,
                    metrics_dir=metrics_dir)
    
    print("[INFO] : Starting training")
    start_time = timer()
    train_model.fit(train_ds, validation_data=val_ds, epochs=cfg.training.epochs, callbacks=callbacks)
    end_time = timer()

    # Log the last epoch history
    last_epoch = log_last_epoch_history(cfg, output_dir)
    
    # Calculate and log the runtime in the log file
    fit_run_time = int(end_time - start_time)
    average_time_per_epoch = round(fit_run_time / (int(last_epoch) + 1),2)
    print("Training runtime: " + str(timedelta(seconds=fit_run_time))) 
    log_to_file(cfg.output_dir, (f"Training runtime : {fit_run_time} s\n" + f"Average time per epoch : {average_time_per_epoch} s"))

    # Set weights and models paths
    best_weights_path = os.path.join(saved_models_dir, "best_weights.h5")
    best_model_path = os.path.join(saved_models_dir, "best_model.h5")
    last_weights_path = os.path.join(saved_models_dir, "last_weights.h5")
    last_model_path = os.path.join(saved_models_dir, "last_model.h5")
    
    print("[INFO] Saved trained models:")
    print("  best model:", best_model_path)
    print("  last model:", last_model_path)
    
    # Save the best model
    base_model.load_weights(best_weights_path)
    base_model.save(best_model_path)
    
    # Save the last model
    base_model.load_weights(last_weights_path)
    base_model.save(last_model_path)

    evaluate(cfg, model_path=best_model_path)

    return best_model_path
