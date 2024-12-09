# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
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
from st_movenet_lightning_heatmaps import st_movenet_lightning_heatmaps
from custom import custom

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
        'heatmaps_spe': None
    }

    model_name = cfg.general.model_type   
    message = "\nPlease check the 'general' section of your configuration file."
    check_model_support(model_name, supported_models=supported_models, message=message)

    cft = cfg.training.model
    input_shape  = cft.input_shape    
    nb_keypoints = cfg.dataset.keypoints
    random_resizing = True if cfg.data_augmentation and cfg.data_augmentation.config.random_periodic_resizing else False
    section = "training.model"
    model = None

    if cft.name == 'st_movenet_lightning_heatmaps':
        check_attributes(cft, expected=["name","alpha","input_shape"], optional=["pretrained_weights"], section=section)
        model = st_movenet_lightning_heatmaps(input_shape=input_shape,
                                              nb_keypoints=nb_keypoints,
                                              alpha=cft.alpha,
                                              pretrained_weights=cft.pretrained_weights)
    elif cft.name == "custom":
        check_attributes(cft, expected=["name","input_shape"], section=section)
        model = custom(input_shape=input_shape,
                       nb_keypoints=nb_keypoints)

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
