# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import numpy as np
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from onnx import ModelProto
import onnxruntime

from models_mgt import model_family, ai_runner_invoke
from models_utils import ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from datasets import get_prediction_data_loader
from postprocess import get_nmsed_detections
from bounding_boxes_utils import bbox_normalized_to_abs_coords, plot_bounding_boxes
from random_utils import remap_pixel_values_range


def view_image_and_boxes(cfg, image, boxes=None, classes=None, scores=None, class_names=None):
        
    # Convert TF tensors to numpy
    image = np.array(image, dtype=np.float32)
    boxes = np.array(boxes, dtype=np.int32)
    classes = np.array(classes, dtype=np.int32)

    print(boxes.shape)

    # Calculate dimensions for the displayed image
    image_width, image_height = np.shape(image)[:2]
    display_size = 7
    if image_width >= image_height:
        x_size = display_size
        y_size = round((image_width / image_height) * display_size)
    else:
        x_size = round((image_height / image_width) * display_size)
        y_size = display_size

    # Display the image and the bounding boxes
    if cfg.general.display_figures:
        fig, ax = plt.subplots(figsize=(x_size, y_size))
        ax.imshow(image)
        plot_bounding_boxes(ax, boxes, classes, scores, class_names)
        plt.show()
        plt.close()
    

def predict_float_model(cfg, model_path):

    print("Loading model file:", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    image_size = model.input.shape[1:3]

    cpr = cfg.preprocessing.rescaling
    pixels_range = (cpr.offset, 255 * cpr.scale + cpr.offset)

    data_loader = get_prediction_data_loader(cfg, image_size=image_size)
    
    cpp = cfg.postprocessing
    for images in data_loader:
        batch_size = tf.shape(images)[0]

        # Predict the images and get the NMS'ed detections
        predictions = model(images)
        boxes, scores, classes = get_nmsed_detections(cfg, predictions, image_size)

        # Display images and boxes
        images = remap_pixel_values_range(images, pixels_range, (0, 1))
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)        
        for i in range(batch_size):
            view_image_and_boxes(cfg, 
                                 images[i],
                                 boxes[i],
                                 classes[i],
                                 scores[i],
                                 class_names=cfg.dataset.class_names)


def predict_quantized_model(cfg, model_path):

    if cfg.prediction and cfg.prediction.target:
        target = cfg.prediction.target
    else:
        target = "host"
    name_model = os.path.basename(model_path)

    print("Loading TFlite model file:", model_path)
    interpreter = tf.lite.Interpreter(model_path)

    ai_runner_interpreter = ai_runner_interp(target,name_model)

    input_details = interpreter.get_input_details()[0]

    batch_size = 1

    input_shape = tuple(input_details['shape'][1:])
    image_size = input_shape[:2]
    
    output_details = interpreter.get_output_details()

    data_loader = get_prediction_data_loader(cfg, image_size=image_size, batch_size=batch_size)

    cpr = cfg.preprocessing.rescaling
    pixels_range = (cpr.offset, 255 * cpr.scale + cpr.offset)
    cpp = cfg.postprocessing
    
    for images in data_loader:
        batch_size = tf.shape(images)[0]

        # Allocate input tensor to predict the batch of images
        input_index = input_details['index']
        tensor_shape = (batch_size,) + input_shape
        interpreter.resize_tensor_input(input_index, tensor_shape)
        interpreter.allocate_tensors()
    
        # Rescale the image using the model's coefficients
        scale = input_details['quantization'][0]
        zero_points = input_details['quantization'][1]
        predict_images = images / scale + zero_points
        
        # Convert the image data type to the model input data type
        # and clip to the min/max values of this data type
        input_dtype = input_details['dtype']
        predict_images = tf.cast(predict_images, input_dtype)
        predict_images = tf.clip_by_value(predict_images, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)

        if target == 'host':
            # Predict the images
            interpreter.set_tensor(input_index, predict_images)
            interpreter.invoke()
        elif target == 'stedgeai_host' or target == 'stedgeai_n6':
            data        = ai_interp_input_quant(ai_runner_interpreter,images.numpy(),cfg.preprocessing.rescaling.scale, cfg.preprocessing.rescaling.offset,'.tflite')
            predictions = ai_runner_invoke(data,ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)

        if len(output_details) == 3:
            if target == 'host':
                # SSD model: outputs are scores, boxes and anchors.
                predictions = (interpreter.get_tensor(output_details[0]['index']),
                               interpreter.get_tensor(output_details[1]['index']),
                               interpreter.get_tensor(output_details[2]['index']))
        else:
            if target == 'host':
                predictions = interpreter.get_tensor(output_details[0]['index'])
            elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                predictions = predictions[0]

        # Decode and NMS the predictions
        boxes, scores, classes = get_nmsed_detections(cfg, predictions, image_size)
 
        # Display images and boxes
        images = remap_pixel_values_range(images, pixels_range, (0, 1))
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)        
        for i in range(batch_size):
            view_image_and_boxes(cfg, 
                                 images[i],
                                 boxes[i],
                                 classes[i],
                                 scores[i],
                                 class_names=cfg.dataset.class_names)


def predict_onnx_model(cfg, model_path, num_classes=None):

    if cfg.prediction and cfg.prediction.target:
        target = cfg.prediction.target
    else:
        target = "host"
    name_model = os.path.basename(model_path)

    print("Loading ONNX model file:", model_path)

    onx = ModelProto()
    with open(model_path, "rb") as f:
        content = f.read()
        onx.ParseFromString(content)
      
    # Get the model input shape (the model is channel first).
    sess = onnxruntime.InferenceSession(model_path)
    input_shape = sess.get_inputs()[0].shape

    ai_runner_interpreter = ai_runner_interp(target,name_model)

    batch_size = 1

    input_shape = (input_shape[2], input_shape[3], input_shape[1])
    image_size = input_shape[:2]

    # Create the data loader
    data_loader = get_prediction_data_loader(cfg, image_size=image_size, batch_size=batch_size)
    
    cpr = cfg.preprocessing.rescaling
    pixels_range = (cpr.offset, 255 * cpr.scale + cpr.offset)

    inputs  = sess.get_inputs()
    outputs = sess.get_outputs()

    for images in data_loader:
        batch_size = tf.shape(images)[0]
        
        channel_first_images = np.transpose(images.numpy(), [0, 3, 1, 2])
        if target == 'host':
            predictions = sess.run([o.name for o in outputs], {inputs[0].name: channel_first_images})
        elif target == 'stedgeai_host' or target == 'stedgeai_n6':
            data        = ai_interp_input_quant(ai_runner_interpreter,channel_first_images,cfg.preprocessing.rescaling.scale,cfg.preprocessing.rescaling.offset,'.onnx')
            predictions = ai_runner_invoke(data,ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)

        if len(predictions) == 1:
            predictions = predictions[0]

        # Decode and NMS the predictions
        boxes, scores, classes = get_nmsed_detections(cfg, predictions, image_size)
                
        # Display images and boxes
        images = remap_pixel_values_range(images, pixels_range, (0, 1))
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)        
        for i in range(batch_size):
            view_image_and_boxes(cfg, 
                                 images[i],
                                 boxes[i],
                                 classes[i],
                                 scores[i],
                                 class_names=cfg.dataset.class_names)
                                 

def predict(cfg):
    """
    Run inference on all the images within the test set.

    Args:
        cfg (config): The configuration file.
    Returns:
        None.
    """

    print("Use ctl+c to exit the script")
    
    model_path = cfg.general.model_path
    
    if Path(model_path).suffix == ".h5":
        predict_float_model(cfg, model_path)
    elif Path(model_path).suffix == ".tflite":
         predict_quantized_model(cfg, model_path)
    elif Path(model_path).suffix == ".onnx":
         predict_onnx_model(cfg, model_path)
    else:
        raise RuntimeError("Evaluation internal error: unsupported model type")
