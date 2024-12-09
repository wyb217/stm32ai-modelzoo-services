# /*---------------------------------------------------------------------------------------------keras
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import shutil
from pathlib import Path
from string import ascii_letters, digits
import random
from timeit import default_timer as timer
from datetime import timedelta
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
from tabulate import tabulate
import math
import mlflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from onnx import ModelProto
import onnxruntime

from bounding_boxes_utils import bbox_normalized_to_abs_coords
from models_mgt import model_family
from datasets import get_evaluation_data_loader
from postprocess import get_nmsed_detections
from objdet_metrics import ObjectDetectionMetricsData, calculate_objdet_metrics, calculate_average_metrics
from logs_utils import log_to_file
from models_utils import count_h5_parameters, ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from models_mgt   import ai_runner_invoke


def evaluate_float_model(cfg: DictConfig, model_path: str, num_classes: int = None) -> dict:

    # Load the model to evaluate
    model = tf.keras.models.load_model(model_path, compile=False)
    
    # Create the data loader
    image_size = model.input.shape[1:3]
    data_loader = get_evaluation_data_loader(cfg, image_size=image_size, normalize=False)

    # Get the size of the dataset
    dataset_size = sum([x.shape[0] for x, _ in data_loader])
    
    # Get the number of groundtruth labels used in the dataset
    _, labels = iter(data_loader).next()
    num_labels = int(tf.shape(labels)[1])
    
    cpp = cfg.postprocessing
    metrics_data = ObjectDetectionMetricsData(num_labels, cpp.max_detection_boxes, dataset_size)

    for data in tqdm(data_loader):
        images, gt_labels = data
        image_size = tf.shape(images)[1:3]

        # Predict the images, decode and NMS the detections
        predictions = model(images)
        boxes, scores, classes = get_nmsed_detections(cfg, predictions, image_size)

        # Record GT boxes and detection boxes in (x1, y1, x2, y2) absolute coordinates
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)
        metrics_data.add_data(gt_labels, boxes, scores, classes)
    
    groundtruths, detections = metrics_data.get_data()
    metrics = calculate_objdet_metrics(groundtruths, detections, cpp.IoU_eval_thresh)

    return metrics


def evaluate_quantized_model(cfg: DictConfig, 
                             model_path: str, 
                             num_classes: int = None,
                             output_dir: str = None) -> dict:

    if cfg.evaluation and cfg.evaluation.target:
        target = cfg.evaluation.target
    else:
        target = "host"
    name_model = os.path.basename(model_path)

    if cfg.general.num_threads_tflite:
        interpreter = tf.lite.Interpreter(model_path, num_threads=cfg.general.num_threads_tflite)
    else:
        interpreter = tf.lite.Interpreter(model_path)

    ai_runner_interpreter = ai_runner_interp(target,name_model)
        
    input_details = interpreter.get_input_details()[0]

    model_batch_size = input_details['shape_signature'][0]
    if model_batch_size!=1 and target == 'host':
        batch_size = 64
    else:
        batch_size = 1

    input_shape = tuple(input_details['shape'][1:])
    image_size = input_shape[:2]

    output_details = interpreter.get_output_details()

    # Create the data loader
    data_loader = get_evaluation_data_loader(cfg, image_size=image_size, normalize=False, batch_size=batch_size)

    # Get the size of the dataset
    dataset_size = sum([x.shape[0] for x, _ in data_loader])
    
    # Get the number of groundtruth labels used in the dataset
    _, labels = iter(data_loader).next()
    num_labels = int(tf.shape(labels)[1])

    cpp = cfg.postprocessing
    metrics_data = ObjectDetectionMetricsData(num_labels, cpp.max_detection_boxes, dataset_size)
    predictions_all = []
    images_full = []

    for data in tqdm(data_loader):
        imag, gt_labels = data
        batch_size = int(tf.shape(imag)[0])

        # Allocate input tensor to predict the batch of images
        input_index = input_details['index']
        tensor_shape = (batch_size,) + input_shape
        
        interpreter.resize_tensor_input(input_index, tensor_shape)
        interpreter.allocate_tensors()
    
        # Rescale the image using the model's coefficients
        scale = input_details['quantization'][0]
        zero_points = input_details['quantization'][1]
        images = imag / scale + zero_points
        
        # Convert the image data type to the model input data type
        # and clip to the min/max values of this data type
        input_dtype = input_details['dtype']
        images = tf.cast(images, input_dtype)
        images = tf.clip_by_value(images, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)                 

        if "evaluation" in cfg and cfg.evaluation:
            if "gen_npy_input" in cfg.evaluation and cfg.evaluation.gen_npy_input==True: 
                images_full.append(images)

        if target == 'host':
            # Predict the images
            interpreter.set_tensor(input_index, images)
            interpreter.invoke()
        elif target == 'stedgeai_host' or target == 'stedgeai_n6':
            data        = ai_interp_input_quant(ai_runner_interpreter,imag.numpy(),cfg.preprocessing.rescaling.scale, cfg.preprocessing.rescaling.offset,'.tflite')
            predictions = ai_runner_invoke(data,ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)

        if model_family(cfg.general.model_type) in ["ssd", "st_yolo_x"]:
            if target == 'host':
                # Model outputs are scores, boxes and anchors.
                predictions = (interpreter.get_tensor(output_details[0]['index']),
                               interpreter.get_tensor(output_details[1]['index']),
                               interpreter.get_tensor(output_details[2]['index']))
        else:
            if target == 'host':
                predictions = interpreter.get_tensor(output_details[0]['index'])
            elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                predictions = predictions[0]

        if "evaluation" in cfg and cfg.evaluation:
            if "gen_npy_output" in cfg.evaluation and cfg.evaluation.gen_npy_output==True:
                predictions_all.append(predictions)

        # Decode and NMS the detections
        boxes, scores, classes = get_nmsed_detections(cfg, predictions, image_size)

        # Record GT boxes and detection boxes in (x1, y1, x2, y2) absolute coordinates
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)
        metrics_data.add_data(gt_labels, boxes, scores, classes)

    # Saves evaluation dataset in a .npy
    if "evaluation" in cfg and cfg.evaluation:
        if "gen_npy_input" in cfg.evaluation and cfg.evaluation.gen_npy_input==True: 
            if "npy_in_name" in cfg.evaluation and cfg.evaluation.npy_in_name:
                npy_in_name = cfg.evaluation.npy_in_name
            else:
                npy_in_name = "unknown_npy_in_name"
            images_full = np.concatenate(images_full, axis=0)
            np.save(os.path.join(output_dir, f"{npy_in_name}.npy"), images_full)

    # Saves model output in a .npy
    if "evaluation" in cfg and cfg.evaluation:
        if "gen_npy_output" in cfg.evaluation and cfg.evaluation.gen_npy_output==True: 
            if "npy_out_name" in cfg.evaluation and cfg.evaluation.npy_out_name:
                npy_out_name = cfg.evaluation.npy_out_name
            else:
                npy_out_name = "unknown_npy_out_name"
            predictions_all = np.concatenate(predictions_all, axis=0)
            np.save(os.path.join(output_dir, f"{npy_out_name}.npy"), predictions_all)

    groundtruths, detections = metrics_data.get_data()
    metrics = calculate_objdet_metrics(groundtruths, detections, cpp.IoU_eval_thresh)

    return metrics    
    

def evaluate_onnx_model(cfg: DictConfig, model_path: str, num_classes: int = None) -> dict:

    if cfg.evaluation and cfg.evaluation.target:
        target = cfg.evaluation.target
    else:
        target = "host"
    name_model = os.path.basename(model_path)

    onx = ModelProto()
    with open(model_path, "rb") as f:
        content = f.read()
        onx.ParseFromString(content)
      
    # Get the model input shape
    sess = onnxruntime.InferenceSession(model_path)
    input_shape = sess.get_inputs()[0].shape

    ai_runner_interpreter = ai_runner_interp(target,name_model)
    
    model_batch_size = input_shape[0]
    if model_batch_size!=1 and target == 'host':
        batch_size = 64
    else:
        batch_size = 1
    
    input_shape = [input_shape[2], input_shape[3], input_shape[1]]
    image_size = input_shape[:2]

    inputs  = sess.get_inputs()
    outputs = sess.get_outputs()

    # Create the data loader
    data_loader = get_evaluation_data_loader(cfg, image_size=image_size, normalize=False, batch_size=batch_size)

    # Get the size of the dataset
    dataset_size = sum([x.shape[0] for x, _ in data_loader])
    
    # Get the number of groundtruth labels used in the dataset
    _, labels = iter(data_loader).next()
    num_labels = int(tf.shape(labels)[1])

    cpp = cfg.postprocessing
    metrics_data = ObjectDetectionMetricsData(num_labels, cpp.max_detection_boxes, dataset_size)

    for data in tqdm(data_loader):
        images, gt_labels = data

        # Predict the images
        images = np.transpose(images.numpy(), [0,3,1,2])
        if target == 'host':
            predictions = sess.run([o.name for o in outputs], {inputs[0].name: images})
        elif target == 'stedgeai_host' or target == 'stedgeai_n6':
            data        = ai_interp_input_quant(ai_runner_interpreter,images,cfg.preprocessing.rescaling.scale,cfg.preprocessing.rescaling.offset,'.onnx')
            predictions = ai_runner_invoke(data,ai_runner_interpreter)
            predictions = ai_interp_outputs_dequant(ai_runner_interpreter,predictions)

        if len(predictions) == 1:
            predictions = predictions[0]

        # Decode and NMS the detections
        boxes, scores, classes = get_nmsed_detections(cfg, predictions, image_size)

        # Record GT boxes and detection boxes in (x1, y1, x2, y2) absolute coordinates
        boxes = bbox_normalized_to_abs_coords(boxes, image_size=image_size)
        metrics_data.add_data(gt_labels, boxes, scores, classes)

    groundtruths, detections = metrics_data.get_data()

    metrics = calculate_objdet_metrics(groundtruths, detections, cpp.IoU_eval_thresh)

    return metrics    
    
    
def display_objdet_metrics(metrics, class_names):
    
    table = []
    classes = list(metrics.keys())    
    for c in sorted(classes):
        table.append([
            class_names[c],
            round(100 * metrics[c].pre, 1),
            round(100 * metrics[c].rec, 1),
            round(100 * metrics[c].ap, 1)])
            
    print()
    headers = ["Class name", "Precision %", "  Recall %", "   AP %  "]
    print()
    print(tabulate(table, headers=headers, tablefmt="pipe", numalign="center"))

    mpre, mrec, mAP = calculate_average_metrics(metrics)
    
    print("\nAverages over classes %:")
    print("-----------------------")
    print(" Mean precision: {:.1f}".format(100 * mpre))
    print(" Mean recall:    {:.1f}".format(100 * mrec))
    print(" Mean AP (mAP):  {:.1f}".format(100 * mAP))


def plot_precision_versus_recall(metrics, class_names, plots_dir):
    """
    Plot the precision versus recall curves. AP values are the areas under these curves.
    """

    # Create the directory where plots will be saved
    if os.path.exists(plots_dir):
        rmtree(plots_dir)
    os.makedirs(plots_dir)

    for c in list(metrics.keys()):
        
        # Plot the precision versus recall curve
        figure = plt.figure(figsize=(10, 10))
        plt.xlabel("recall")
        plt.ylabel("interpolated precision")
        plt.title("Class '{}' (AP = {:.2f})".
                    format(class_names[c], metrics[c].ap * 100))
        plt.plot(metrics[c].interpolated_precision, metrics[c].interpolated_recall)
        plt.grid()

        # Save the plot in the plots directory
        plt.savefig(f"{plots_dir}/{class_names[c]}.png")
        plt.close(figure)


def evaluate(cfg: DictConfig, model_path: str = None):
    
    output_dir = HydraConfig.get().runtime.output_dir
    if not model_path:
        model_path = cfg.general.model_path

    cpp = cfg.postprocessing
    print("Metrics calculation parameters:")
    print("  confidence threshold:", cpp.confidence_thresh)
    print("  NMS IoU threshold:", cpp.NMS_thresh)
    print("  max detection boxes:", cpp.max_detection_boxes)
    print("  metrics IoU threshold:", cpp.IoU_eval_thresh)

    model_type = "float" if Path(model_path).suffix == ".h5" else "quantized"
    print(f"Evaluating {model_type} model: {model_path}")

    class_names = cfg.dataset.class_names
    num_classes = len(class_names)

    start_time = timer()
    if Path(model_path).suffix == '.h5':
        count_h5_parameters(output_dir=output_dir, 
                            model_path=model_path)
        metrics = evaluate_float_model(cfg, model_path, num_classes=num_classes)
    elif Path(model_path).suffix == '.tflite':
        metrics = evaluate_quantized_model(cfg, model_path, num_classes=num_classes, output_dir=output_dir)
    elif Path(model_path).suffix == '.onnx':
        metrics = evaluate_onnx_model(cfg, model_path, num_classes=num_classes)
    else:
        raise RuntimeError("Evaluation internal error: unsupported model "
                           f"file extension {model_path}")

    end_time = timer()
    eval_run_time = int(end_time - start_time)
    print("Evaluation run time: " + str(timedelta(seconds=eval_run_time)))

    display_objdet_metrics(metrics, class_names)
            
    # Log metrics in the stm32ai_main.log file
    log_to_file(output_dir, f"{model_type} model dataset used: {cfg.dataset.name}")
    
    mpre, mrec, mAP = calculate_average_metrics(metrics)
    
    log_to_file(output_dir, "{}_model_mpre: {:.1f}".format(model_type, 100 * mpre))
    log_to_file(output_dir, "{}_model_mrec: {:.1f}".format(model_type, 100 * mrec))
    log_to_file(output_dir, "{}_model_map: {:.1f}".format(model_type, 100 * mAP))
    
    # Log metrics in mlflow
    mlflow.log_metric(f"{model_type}_model_mpre", 100 * mpre)
    mlflow.log_metric(f"{model_type}_model_mrec", 100 * mrec)
    mlflow.log_metric(f"{model_type}_model_mAP", 100 * mAP)

    if cfg.postprocessing.plot_metrics:
        print("\nPlotting precision versus recall curves")
        plots_dir = os.path.join(output_dir, "precision_vs_recall_curves", os.path.basename(model_path))
        print("Plots directory:", plots_dir)
        
        output_dir = HydraConfig.get().runtime.output_dir
        model_path_suffix = Path(model_path).suffix
        plot_precision_versus_recall(metrics, class_names, plots_dir)

