# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import sys
import tqdm
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import numpy as np
import tensorflow as tf
from onnx import ModelProto
import onnxruntime
from typing import Optional

from datasets import get_quantization_data_loader
from onnx_quantizer import quantize_onnx
from onnx_evaluation import model_is_quantized
from models_utils import tf_dataset_to_np_array
from onnx_model_convertor import onnx_model_converter
from model_formatting_ptq_per_tensor import model_formatting_ptq_per_tensor


def tflite_ptq_quantizer(model: tf.keras.Model = None,
                         quantization_ds: tf.data.Dataset = None,
                         output_dir: str = None,
                         export_dir: Optional[str] = None,
                         input_shape: tuple = None,
                         rescaling: tuple = None,
                         quantization_granularity: str = None,
                         quantization_input_type: str = None,
                         quantization_output_type: str = None) -> None:
                             
    """
    Perform post-training quantization on a TensorFlow Lite model.

    Args:
        model (tf.keras.Model): The TensorFlow model to be quantized.
        quantization_ds (tf.data.Dataset): The quantization dataset if it's provided by the user else the training
        dataset. Defaults to None
        rescaling (tuple): input rescaling parameters, [0]: scaling, [1]: offset
        output_dir (str): Path to the output directory. Defaults to None.
        export_dir (str): Name of the export directory. Defaults to None.
        input_shape (tuple: The input shape of the model. Defaults to None.
        quantization_granularity (str): 'per_tensor' or 'per_channel'. Defaults to None.
        quantization_input_type (str): The quantization type for the input. Defaults to None.
        quantization_output_type (str): The quantization type for the output. Defaults to None.

    Returns:
        None
    """

    def data_gen():
        """
        Generate data for post-training quantization.

        Yields:
            List[tf.Tensor]: A list of TensorFlow tensors representing the input data.
        """
        if not quantization_ds:
            for _ in tqdm.tqdm(range(5)):
                image = tf.random.uniform((1,) + input_shape, minval=0, maxval=256, dtype=tf.int32)
                image = tf.cast(image, tf.float32)
                image = rescaling[0] * image + rescaling[1]
                yield [image]
        else:
            for images in tqdm.tqdm(quantization_ds, total=len(quantization_ds)):
                yield [images]

    # Create the TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Create the output directory
    tflite_models_dir = Path(os.path.join(output_dir, "{}/".format(export_dir)))
    tflite_models_dir.mkdir(exist_ok=True, parents=True)

    # Set the quantization types for the input and output
    if quantization_input_type == 'int8':
        converter.inference_input_type = tf.int8
    elif quantization_input_type == 'uint8':
        converter.inference_input_type = tf.uint8

    if quantization_output_type == 'int8':
        converter.inference_output_type = tf.int8
    elif quantization_output_type == 'uint8':
        converter.inference_output_type = tf.uint8

    # Set the optimizations and representative dataset generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = data_gen

    if quantization_granularity == 'per_tensor':
        converter._experimental_disable_per_channel = True

    # Convert the model to a quantized TFLite model
    tflite_model_quantized = converter.convert()

    # Save the quantized model
    tflite_model_quantized_file = tflite_models_dir / "quantized_model.tflite"
    tflite_model_quantized_file.write_bytes(tflite_model_quantized)


def quantize_keras_model(cfg: DictConfig, model_path):
    """
    Quantize a Keras model using TFlite

    Args:
        cfg (DictConfig): Entire configuration file.
        model_path: Optional[str]: Path to the float model

    Returns:
        quantized model path
    """

    # Create the export directory if it does not exist
    output_dir = HydraConfig.get().runtime.output_dir
    export_dir = cfg.quantization.export_dir

    print("Loading float model file:", model_path)
    float_model = tf.keras.models.load_model(model_path, compile=False)
    input_shape = float_model.input.shape[1:]

    # Get the data loader
    quantization_ds = get_quantization_data_loader(cfg, image_size=input_shape[:2], batch_size=1)
    
    if not quantization_ds:
        print("[WARNING] No representative images were provided. Quantization will be performed using fake data.")

    cfq = cfg.quantization
    quantization_granularity = cfg.quantization.granularity
    quantization_optimize = cfg.quantization.optimize
    print(f'[INFO] : Quantization granularity : {quantization_granularity}')
    
    # if per-tensor quantization is required some optimizations are possible on the float model
    if quantization_granularity == 'per_tensor' and quantization_optimize:
        print("[INFO] : Optimizing the model for improved per_tensor quantization...")
        float_model = model_formatting_ptq_per_tensor(model_origin=float_model)
        model_path = os.path.join(Path(output_dir), cfq.export_dir, "optimized_model.h5")
        float_model.save(model_path)

    cpp = cfg.preprocessing.rescaling

    if cfq.quantizer.lower() == "tflite_converter" and cfq.quantization_type == "PTQ":
        tflite_ptq_quantizer(float_model,
                             quantization_ds=quantization_ds,
                             output_dir=output_dir,
                             export_dir=export_dir,
                             input_shape=input_shape,
                             rescaling=(cpp.scale, cpp.offset),
                             quantization_granularity=cfq.granularity,
                             quantization_input_type=cfq.quantization_input_type,
                             quantization_output_type=cfq.quantization_output_type)
        quantized_model_path = os.path.join(output_dir, export_dir, "quantized_model.tflite")
        print("Quantized model path:", quantized_model_path)
        return quantized_model_path
    elif cfq.quantizer.lower() == "onnx_quantizer" and cfq.quantization_type == "PTQ":
        # Convert the dataset to numpy array
        target_opset = cfg.quantization.target_opset
        if quantization_ds:
            data, _ = tf_dataset_to_np_array(quantization_ds, labels_included=False, nchw=True)
        else:
            data = None
        converted_model_path = os.path.join(output_dir, 'converted_model', 'converted_model.onnx')
        onnx_model_converter(input_model_path=model_path, target_opset=target_opset, output_dir=converted_model_path)
        cfg.general.model_path = converted_model_path
        quantized_model_path = quantize_onnx(quantization_samples=data, configs=cfg)
        print("Quantized model path:", quantized_model_path)
        return quantized_model_path
    else:
        raise TypeError("Quantizer or quantization type not supported. Check the `quantization` section of your "
                        "user_config.yaml file!")


def quantize_onnx_model(cfg, model_path):
    """
        Quantize a onnx model using ONNX runtime

        Args:
            cfg (DictConfig): Entire configuration file.
            model_path: Optional[str]: Path to the float model

        Returns:
            quantized model path
        """

    # Get the model input shape
    onx = ModelProto()
    with open(model_path, "rb") as f:
        content = f.read()
        onx.ParseFromString(content)
    sess = onnxruntime.InferenceSession(model_path)
    input_shape = sess.get_inputs()[0].shape
    input_shape = input_shape[-3:]
    
    # ONNX models are channel first. We need channel last for the data loader.
    input_shape = (input_shape[1], input_shape[2], input_shape[0])
    
    # Get the representative images    
    quantization_ds = get_quantization_data_loader(cfg, image_size=input_shape[:2], batch_size=1)

    if quantization_ds:
        data, _ = tf_dataset_to_np_array(quantization_ds, labels_included=False, nchw=True)
    else:
        print("[WARNING] No representative images were provided. Quantization will be performed using fake data.")
        data = None

    # Quantize the model
    cfq = cfg.quantization
    
    if model_is_quantized(model_path):
        print('[INFO]: The input model is already quantized!\n\tReturning the same model!')
        return model_path
    elif cfq.quantizer.lower() == "onnx_quantizer" and cfq.quantization_type == "PTQ":
        # Quantize the model
        cfg.general.model_path = model_path
        quantized_model_path = quantize_onnx(quantization_samples=data, configs=cfg)
    else:
        raise TypeError("Quantizer or quantization type not supported. Check the `quantization` section of your "
                        "user_config.yaml file!")
    
    return quantized_model_path
    

def quantize(cfg, model_path=None):

    model_path = model_path if model_path else cfg.general.model_path

    # Check the provided quantizer name
    cfq = cfg.quantization

    if Path(model_path).suffix == ".onnx":
        if cfq.quantizer.lower() != "onnx_quantizer":
            raise ValueError("\nUnknown or unsupported quantizer\n"
                             f"Received: {cfq.quantizer}\n"
                             "Supported quantizer: onnx_quantizer")

    print("[INFO] : Quantizing the model ... This might take few minutes ...")
    start_time = timer()
    if Path(model_path).suffix == ".h5":
        quantized_model_path = quantize_keras_model(cfg, model_path)
    elif Path(model_path).suffix == ".onnx":
        quantized_model_path = quantize_onnx_model(cfg, model_path)
    end_time = timer()
    
    run_time = int(end_time - start_time)
    print("Quantization runtime: " + str(timedelta(seconds=run_time))) 

    return quantized_model_path
