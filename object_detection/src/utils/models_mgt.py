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
import tensorflow as tf
from onnx import ModelProto
import onnxruntime
from omegaconf import DictConfig
import numpy as np

from cfg_utils import check_attributes
from models_utils import check_model_support, check_attribute_value 
from st_ssd_mobilenet_v1 import st_ssd_mobilenet_v1
from ssd_mobilenet_v2_fpnlite import ssd_mobilenet_v2_fpnlite
from tiny_yolo_v2 import tiny_yolo_v2
from st_yolo_lc_v1 import st_yolo_lc_v1
from st_yolo_x import st_yolo_x

def ai_runner_invoke(image_processed,ai_runner_interpreter):
    def reduce_shape(x):  # reduce shape (request by legacy API)
        old_shape = x.shape
        n_shape = [old_shape[0]]
        for v in x.shape[1:len(x.shape) - 1]:
            if v != 1:
                n_shape.append(v)
        n_shape.append(old_shape[-1])
        return x.reshape(n_shape)

    preds, _ = ai_runner_interpreter.invoke(image_processed)
    predictions = []
    for x in preds:
        x = reduce_shape(x)
        predictions.append(x.copy())
    return predictions

def model_family(model_type: str) -> str:
    if model_type in ("st_ssd_mobilenet_v1", "ssd_mobilenet_v2_fpnlite"):
        return "ssd"
    elif model_type in ("tiny_yolo_v2", "st_yolo_lc_v1"):
        return "yolo"
    elif model_type in ("yolo_v8", "yolo_v5u"):
        return "yolo_v8"
    elif model_type in ("st_yolo_x"):
        return "st_yolo_x"
    else:
        raise ValueError(f"Internal error: unknown model type {model_type}")


def check_ssd_mobilenet(cft, model_type, alpha_values=None, random_resizing=None):

    check_attributes(cft, expected=["alpha", "input_shape"], optional=["pretrained_weights"], section="training.model")
                          
    message = "\nPlease check the 'training.model' section of your configuration file."
    if cft.alpha not in alpha_values:
        raise ValueError(f"\nSupported `alpha` values for `{model_type}` model are "
                         f"{alpha_values}. Received {cft.alpha}{message}")
                         
    if random_resizing:
        raise ValueError(f"\nrandom_periodic_resizing is not supported for model `{model_type}`.\n"
                         "Please check the 'data_augmentation' section of your configuration file.")

def check_st_yolo_x(cft, model_type, random_resizing=None):

    check_attributes(cft, expected=["input_shape"], optional=["depth_mul", "width_mul"], section="training.model")
                          
    message = "\nPlease check the 'training.model' section of your configuration file."                        
    if random_resizing:
        raise ValueError(f"\nrandom_periodic_resizing is not supported for model `{model_type}`.\n"
                         "Please check the 'data_augmentation' section of your configuration file.")


def get_zoo_model(cfg: DictConfig):
    """
    Returns a Keras model object based on the specified configuration and parameters.

    Args:
        cfg (DictConfig): A dictionary containing the configuration for the model.
        num_classes (int): The number of classes for the model.
        dropout (float): The dropout rate for the model.
        section (str): The section of the model to be used.

    Returns:
        tf.keras.Model: A Keras model object based on the specified configuration and parameters.
    """

    # Define the supported models and their versions
    supported_models = {
        'st_ssd_mobilenet_v1': None,
        'ssd_mobilenet_v2_fpnlite': None,
        'tiny_yolo_v2': None,
        'st_yolo_lc_v1': None,
        'st_yolo_x': None,

    }

    model_name = cfg.general.model_type   
    message = "\nPlease check the 'general' section of your configuration file."
    check_model_support(model_name, supported_models=supported_models, message=message)

    cft = cfg.training.model
    input_shape = cft.input_shape    
    num_classes = len(cfg.dataset.class_names)
    random_resizing = True if cfg.data_augmentation and cfg.data_augmentation.config.random_periodic_resizing else False
    section = "training.model"
    model = None

    if model_name == "st_ssd_mobilenet_v1":
        check_ssd_mobilenet(cft, "st_ssd_mobilenet_v1",
                            alpha_values=[0.25, 0.50, 0.75, 1.0],
                            random_resizing=random_resizing)
        model = st_ssd_mobilenet_v1(input_shape, num_classes, cft.alpha, pretrained_weights=cft.pretrained_weights)
        
    elif model_name == "ssd_mobilenet_v2_fpnlite":
        check_ssd_mobilenet(cft, "ssd_mobilenet_v2_fpnlite",
                            alpha_values=[0.35, 0.50, 0.75, 1.0],
                            random_resizing=random_resizing)
        model = ssd_mobilenet_v2_fpnlite(input_shape, num_classes, cft.alpha, pretrained_weights=cft.pretrained_weights)

    elif model_name == "tiny_yolo_v2":     
        check_attributes(cft, expected=["input_shape"], section=section)
        num_anchors = len(cfg.postprocessing.yolo_anchors)
        model = tiny_yolo_v2(input_shape, num_anchors, num_classes)

    elif model_name == "st_yolo_lc_v1":
        check_attributes(cft, expected=["input_shape"], section=section)
        num_anchors = len(cfg.postprocessing.yolo_anchors)
        model = st_yolo_lc_v1(input_shape, num_anchors, num_classes)

    elif model_name == "st_yolo_x":
        check_st_yolo_x(cft, "st_yolo_x",random_resizing=random_resizing)
        num_anchors = len(cfg.postprocessing.yolo_anchors)
        if not cft.depth_mul and not cft.width_mul:
            cft.depth_mul = 0.33
            cft.width_mul = 0.25
        model = st_yolo_x(input_shape=input_shape, num_anchors=num_anchors, num_classes=num_classes, depth_mul=cft.depth_mul, width_mul=cft.width_mul)

    return model
    

def load_model_for_training(cfg: DictConfig) -> tuple:
    """"
    Loads a model for training.
    
    The model to train can be:
    - a model from the Model Zoo
    - a user model (BYOM)
    - a model previously trained during a training that was interrupted.
    
    When a training is run, the following files are saved in the saved_models
    directory:
        base_model.h5:
            Model saved before the training started. Weights are random.
        best_weights.h5:
            Best weights obtained since the beginning of the training.
        last_weights.h5:
            Weights saved at the end of the last epoch.
    
    To resume a training, the last weights are loaded into the base model.
    """
    
    model_type = cfg.general.model_type    
    model = None
    
    # Train a model from the Model Zoo
    if cfg.training.model:
        print("[INFO] : Loading Model Zoo model:", model_type)        
        model = get_zoo_model(cfg)
        
        cft = cfg.training.model
        if cft.pretrained_weights:
            print(f"[INFO] : Loaded pretrained weights: `{cft.pretrained_weights}`")
        else:
            print(f"[INFO] : No pretrained weights were loaded.")
        
    # Bring your own model
    elif cfg.general.model_path:
        print("[INFO] : Loading model", cfg.general.model_path)
        model = tf.keras.models.load_model(cfg.general.model_path, compile=False)
        
        # Check that the model has a specified input shape
        input_shape = tuple(model.input.shape[1:])
        if None in input_shape:
            raise ValueError(f"\nThe model input shape is unspecified. Got {str(input_shape)}\n"
                             "Unable to proceed with training.")
                        
    # Resume a previously interrupted training
    elif cfg.training.resume_training_from:
        resume_dir = os.path.join(cfg.training.resume_training_from, cfg.general.saved_models_dir)
        print(f"[INFO] : Resuming training from directory {resume_dir}\n")
        
        message = "\nUnable to resume training."
        if not os.path.isdir(resume_dir):
            raise FileNotFoundError(f"\nCould not find resume directory {resume_dir}{message}")
        model_path = os.path.join(resume_dir, "base_model.h5")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"\nCould not find model file {model_path}{message}\n")
        last_weights_path = os.path.join(resume_dir, "last_weights.h5")
        if not os.path.isfile(last_weights_path):
            raise FileNotFoundError(f"\nCould not find model weights file {last_weights_path}{message}\n")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        model.load_weights(last_weights_path)

    return model
