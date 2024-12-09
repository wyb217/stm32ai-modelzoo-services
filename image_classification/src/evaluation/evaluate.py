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
from pathlib import Path
import warnings
import sklearn
import mlflow
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from typing import Tuple, Optional, List, Dict
import numpy as np

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import onnxruntime
import tensorflow as tf
import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

from preprocess import apply_rescaling, postprocess_output, preprocess_input
from visualize_utils import plot_confusion_matrix
from logs_utils import log_to_file
from onnx_evaluation import model_is_quantized, predict_onnx, count_onnx_parameters
from models_utils import tf_dataset_to_np_array, compute_confusion_matrix, count_h5_parameters, \
                         count_tflite_parameters, ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from models_mgt import get_loss, ai_runner_invoke


def evaluate_tflite_quantized_model(cfg: DictConfig = None,
                                    quantized_model_path: str = None, eval_ds: tf.data.Dataset = None,
                                    class_names: list = None, output_dir: str = None,
                                    name_ds: Optional[str] = 'test_set',
                                    num_threads: Optional[int] = 1,
                                    display_figures: bool = None) -> float:
    """
    Evaluates the accuracy of a quantized TensorFlow Lite model using tflite.interpreter and plots the confusion matrix.

    Args:
        cfg (config): The configuration file.
        quantized_model_path (str): The file path to the quantized TensorFlow Lite model.
        eval_ds (tf.data.Dataset): The test dataset to evaluate the model on.
        class_names (list): A list of class names for the confusion matrix.
        output_dir (str): The directory where to save the image.
        name_ds (str): The name of the chosen eval_ds to be mentioned in the prints and figures.
        num_threads (int): number of threads for the tflite interpreter
    Returns:
        float: The accuracy of the quantized model.
    """
    tf.print(f'[INFO] : Evaluating the quantized model using {name_ds}...')
    if cfg.evaluation and cfg.evaluation.target:
        target = cfg.evaluation.target
    else:
        target = "host"
    name_model = os.path.basename(quantized_model_path)
    ai_runner_interpreter = ai_runner_interp(target,name_model)
    
    interpreter_quant = tf.lite.Interpreter(model_path=quantized_model_path, num_threads=num_threads)
    interpreter_quant.allocate_tensors()
    input_details = interpreter_quant.get_input_details()[0]
    input_index_quant = input_details["index"]
    output_index_quant = interpreter_quant.get_output_details()[0]["index"]
    output_details = interpreter_quant.get_output_details()[0]
    predictions_all = []
    test_pred = []
    test_labels = []
    images_full = []

    for images, labels in tqdm.tqdm(eval_ds, total=len(eval_ds)):
        for image, label in zip(images, labels):
            image_processed = preprocess_input(image, input_details)
            if "evaluation" in cfg and cfg.evaluation:
                if "gen_npy_input" in cfg.evaluation and cfg.evaluation.gen_npy_input == True:
                    images_full.append(image_processed)
            if target == 'host':
                interpreter_quant.set_tensor(input_index_quant, image_processed)
                interpreter_quant.invoke()
                test_pred_score = interpreter_quant.get_tensor(output_index_quant)
            elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                image_preproc = ai_interp_input_quant(ai_runner_interpreter, image[None].numpy(),
                                                      cfg.preprocessing.rescaling.scale,
                                                      cfg.preprocessing.rescaling.offset,
                                                      '.tflite')
                test_pred_score = ai_runner_invoke(image_preproc, ai_runner_interpreter)
                test_pred_score = ai_interp_outputs_dequant(ai_runner_interpreter, [test_pred_score])[0]
                test_pred_score = np.reshape(test_pred_score, [1, -1])

            if "evaluation" in cfg and cfg.evaluation:
                if "gen_npy_output" in cfg.evaluation and cfg.evaluation.gen_npy_output == True:
                    predictions_all.append(test_pred_score)

            predicted_label = postprocess_output(test_pred_score, output_details)
            test_pred.append(predicted_label)
            test_labels.append(label.numpy())

    # Saves evaluation dataset in a .npy
    if "evaluation" in cfg and cfg.evaluation:
        if "gen_npy_input" in cfg.evaluation and cfg.evaluation.gen_npy_input == True:
            if "npy_in_name" in cfg.evaluation and cfg.evaluation.npy_in_name:
                npy_in_name = cfg.evaluation.npy_in_name
            else:
                npy_in_name = "unknown_npy_in_name"
            images_full = np.concatenate(images_full, axis=0)
            print("[INFO] : Shape of npy input dataset = {}".format(images_full.shape))
            np.save(os.path.join(output_dir, f"{npy_in_name}.npy"), images_full)

    # Saves model output in a .npy
    if "evaluation" in cfg and cfg.evaluation:
        if "gen_npy_output" in cfg.evaluation and cfg.evaluation.gen_npy_output == True:
            if "npy_out_name" in cfg.evaluation and cfg.evaluation.npy_out_name:
                npy_out_name = cfg.evaluation.npy_out_name
            else:
                npy_out_name = "unknown_npy_out_name"
            predictions_all = np.concatenate(predictions_all, axis=0)
            print("[INFO] : Shape of npy predicted scores = {}".format(predictions_all.shape))
            np.save(os.path.join(output_dir, f"{npy_out_name}.npy"), predictions_all)
    
    labels = np.array(test_labels)
    logits = np.concatenate(test_pred, axis=0)
    logits = np.squeeze(logits)
    cm = sklearn.metrics.confusion_matrix(labels, logits)
    accuracy = round((np.sum(labels == logits) * 100) / len(test_labels), 2)

    print(f"[INFO] : Accuracy of quantized model on {name_ds} = {accuracy}%")
    log_to_file(output_dir,  f"Quantized model {name_ds}:")
    log_to_file(output_dir, f"Accuracy of quantized model : {accuracy} %")
    mlflow.log_metric(f"int_acc_{name_ds}", accuracy)
    
    if display_figures:
        model_name = f"quantized_model_confusion_matrix_{name_ds}"
        plot_confusion_matrix(cm, class_names=class_names, model_name=model_name,
                              title=f'{model_name}\naccuracy: {accuracy}', output_dir=output_dir)
    return accuracy


def evaluate_h5_model(model_path: str = None, eval_ds: tf.data.Dataset = None, class_names: list = None,
                      output_dir: str = None, name_ds: Optional[str] = 'test_set',
                      display_figures: bool = None) -> float:
    """
    Evaluates a trained Keras model saved in .h5 format on the provided test data.

    Args:
        model_path (str): The file path to the .h5 model.
        eval_ds (tf.data.Dataset): The test data to evaluate the model on.
        class_names (list): A list of class names for the confusion matrix.
        output_dir (str): The directory where to save the image.
        name_ds (str): The name of the chosen eval_ds to be mentioned in the prints and figures.
    Returns:
        float: The accuracy of the model on the test data.
    """

    # Load the .h5 model
    model = tf.keras.models.load_model(model_path)
    loss = get_loss(num_classes=len(class_names))
    model.compile(loss=loss, metrics=['accuracy'])

    # Evaluate the model on the test data
    tf.print(f'[INFO] : Evaluating the float model using {name_ds}...')
    loss, accuracy = model.evaluate(eval_ds)
    # compute confusion matrix
    cm, accuracy = compute_confusion_matrix(test_set=eval_ds, model=model)

    if display_figures:
        model_name = f"float_model_confusion_matrix_{name_ds}"
        plot_confusion_matrix(cm=cm,
                              class_names=class_names,
                              model_name=model_name,
                              title=f'{model_name}\naccuracy: {accuracy}',
                              output_dir=output_dir)
    
    print(f"[INFO] : Accuracy of float model on {name_ds} = {accuracy}%")
    print(f"[INFO] : Loss of float model on {name_ds} = {loss}")
    mlflow.log_metric(f"float_acc_{name_ds}", accuracy)
    mlflow.log_metric(f"float_loss_{name_ds}", loss)
    log_to_file(output_dir, f"Float model {name_ds}:")
    log_to_file(output_dir, f"Accuracy of float model : {accuracy} %")
    log_to_file(output_dir, f"Loss of float model : {round(loss,2)} ")
    return accuracy


def evaluate_onnx_model(cfg: DictConfig,
                        input_samples: np.ndarray,
                        input_labels: np.ndarray,
                        input_model_path: str,
                        class_labels: List[str],
                        output_dir: str,
                        name_ds: str,
                        display_figures: bool = None) -> Tuple[float, np.ndarray]:

    """
    Evaluates an ONNX model on a validation dataset.

    Args:
        cfg (DictConfig): dict containing all yaml parameters.
        input_samples (np.ndarray): The evaluation samples.
        input_labels (np.ndarray): The labels for the evaluation samples.
        input_model_path (str): The path to the onnx model to be evaluated.
        class_labels (List[str]): The list of the class labels.
        output_dir(str): The name of the output directory where confusion matrix and the logs are to be saved.
        name_ds (str): name of the set on which we evaluate

    Returns:
        Tuple[float, np.ndarray]: The validation accuracy and confusion matrix.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if cfg.evaluation and cfg.evaluation.target:
        target = cfg.evaluation.target
    else:
        target = "host"
    name_model = os.path.basename(input_model_path)
    ai_runner_interpreter = ai_runner_interp(target, name_model)

    sess = onnxruntime.InferenceSession(input_model_path)
    model_type = 'quantized' if model_is_quantized(input_model_path) else 'float'
    if model_type == 'float' or target == 'host':
        prd_labels = predict_onnx(sess, input_samples).argmax(axis=1)
    elif (target == 'stedgeai_host' or target == 'stedgeai_n6') and model_type == 'quantized':
        prd_labels = []
        for i in tqdm.tqdm(range(input_samples.shape[0])):
            data = ai_interp_input_quant(ai_runner_interpreter, input_samples[i][None],
                                         cfg.preprocessing.rescaling.scale,
                                         cfg.preprocessing.rescaling.offset,
                                         '.onnx')
            prd_label = ai_runner_invoke(data, ai_runner_interpreter)
            prd_label = ai_interp_outputs_dequant(ai_runner_interpreter, [prd_label])[0]
            prd_label = prd_label.argmax(axis=1)
            prd_labels.append(prd_label)
        prd_labels = np.array(prd_labels, dtype=np.float32)
    else:
        raise TypeError("Only supported targets are \"host\", \"stedgeai_host\" or \"stedgeai_n6\". "
                        "Check the \"evaluation\" section of your configuration file.")
    accuracy = round(accuracy_score(input_labels, prd_labels) * 100, 2)
    print(f'[INFO] : Evaluation accuracy on {name_ds}: {accuracy} %')
    log_file_name = f"{output_dir}/stm32ai_main.log"
    with open(log_file_name, 'a', encoding='utf-8') as f:
        f.write(f'{model_type} onnx model\nEvaluation accuracy: {accuracy} %\n')
    cm = confusion_matrix(input_labels, prd_labels)
    acc_metric_name = f"int_acc_{name_ds}" if model_is_quantized(input_model_path) else f"float_acc_{name_ds}"
    mlflow.log_metric(acc_metric_name, accuracy)
    
    if display_figures:
        model_name = f'{model_type}_onnx_model_{name_ds}'
        plot_confusion_matrix(cm=cm,
                              class_names=class_labels,
                              model_name=model_name,
                              title=f'{model_name}\naccuracy: {accuracy}', output_dir=output_dir)

    return accuracy, cm


def evaluate(cfg: DictConfig = None, eval_ds: tf.data.Dataset = None,
             model_path_to_evaluate: Optional[str] = None, name_ds: Optional[str] = 'test_set') -> None:
    """
    Evaluates and benchmarks a TensorFlow Lite or Keras model, and generates a Config header file if specified.

    Args:
        cfg (config): The configuration file.
        eval_ds (tf.data.Dataset): The validation dataset.
        model_path_to_evaluate (str, optional): Model path to evaluate
        name_ds (str): The name of the chosen test_data to be mentioned in the prints and figures.

    Returns:
        None
    """
    output_dir = HydraConfig.get().runtime.output_dir
    class_names = cfg.dataset.class_names
    model_path = model_path_to_evaluate if model_path_to_evaluate else cfg.general.model_path
    file_extension = Path(model_path).suffix

    # Pre-process test dataset
    eval_ds = apply_rescaling(dataset=eval_ds, scale=cfg.preprocessing.rescaling.scale,
                              offset=cfg.preprocessing.rescaling.offset)

    try:
        # Check if the model is a TensorFlow Lite model
        if file_extension == '.h5':
            count_h5_parameters(output_dir=output_dir, 
                                model_path=model_path)
            # Evaluate Keras model
            evaluate_h5_model(model_path=model_path, eval_ds=eval_ds,
                              class_names=class_names, output_dir=output_dir, name_ds=name_ds,
                              display_figures=cfg.general.display_figures)
        elif file_extension == '.tflite':
            # Evaluate quantized TensorFlow Lite model
            evaluate_tflite_quantized_model(cfg=cfg,
                                            quantized_model_path=model_path,
                                            eval_ds=eval_ds, class_names=class_names,
                                            output_dir=output_dir, name_ds=name_ds,
                                            num_threads=cfg.general.num_threads_tflite,
                                            display_figures=cfg.general.display_figures)
        elif file_extension == '.onnx':
            # Evaluate quantized or float ONNX model
            data, labels = tf_dataset_to_np_array(eval_ds)
            evaluate_onnx_model(cfg=cfg,
                                input_samples=data,
                                input_labels=labels,
                                input_model_path=model_path,
                                class_labels=class_names,
                                output_dir=output_dir,
                                name_ds=name_ds,
                                display_figures=cfg.general.display_figures)
    except Exception:
        raise ValueError(f"Model accuracy evaluation failed\nReceived model path: {model_path}")
