# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import cv2
from pathlib import Path
import numpy as np
import tensorflow as tf
from typing import Optional, List
from omegaconf import DictConfig
from utils import multiply_tensors, cxcywh_to_xyxy, custom_draw, PaletteManager
from preprocess import preprocess_image, preprocess_input, read_image


def postprocess(image: np.ndarray, masks: np.ndarray, detections: np.ndarray, output_index_quant: List[dict],
                conf_threshold: float, iou_threshold: float, n_masks: int, width: int, height: int, 
                prediction_result_dir: str, file: str, class_names: str) -> None:
    """
    Post-process the predictions and save the results.

    Args:
        image (np.ndarray): The original image.
        masks (np.ndarray): The predicted masks.
        detections (np.ndarray): The predicted detections.
        output_index_quant (List[dict]): Output index quantization details.
        conf_threshold (float): Confidence threshold.
        iou_threshold (float): IoU threshold.
        n_masks (int): Number of masks.
        width (int): Width of the image.
        height (int): Height of the image.
        prediction_result_dir (str): Directory to save the result image.
        file (str): File name for the result image.

    Returns:
        None
    """
    # Dequantize detections
    detections_scale, detections_zero_point = output_index_quant[0]['quantization']
    detections_deq = (detections.astype(np.float32) - detections_zero_point) * detections_scale
    detections_t = np.transpose(detections_deq, (0, 2, 1))

    # Dequantize masks
    masks_scale, masks_zero_point = output_index_quant[1]['quantization']
    masks_deq = (masks.astype(np.float32) - masks_zero_point) * masks_scale
    
    # Filter detections by score
    detections = detections_t[np.amax(detections_t[..., 4:-n_masks], axis=-1) > conf_threshold]
    # Scale normalized box to model width and height
    detections[..., [0, 2]] *= width
    detections[..., [1, 3]] *= height

    # Convert raw detections to final detections structure
    detections = np.c_[detections[..., :4], np.amax(detections[..., 4:-n_masks], axis=-1),
                       np.argmax(detections[..., 4:-n_masks], axis=-1), detections[..., -n_masks:]]
    
    # Apply NMS
    nmsed_detections = detections[cv2.dnn.NMSBoxes(detections[:, :4].tolist(), detections[:, 4].tolist(), 
                                                   conf_threshold, iou_threshold)]

    if nmsed_detections.shape[0] > 0:
        # Transpose masks
        masks_t = np.transpose(masks_deq, (0, 3, 1, 2))

        # Flatten the masks
        squeezed_masks = np.squeeze(masks_t)
        flattened_masks = squeezed_masks.reshape((squeezed_masks.shape[0], -1))

        # Matrix multiplication between detection masks and mask_buffer
        detections_mask = nmsed_detections[:, 6:] 
        post_processed_masks = multiply_tensors(detections_mask, flattened_masks)

        # Restore masks initial shape
        b, c, mh, mw = masks_t.shape
        post_processed_masks = post_processed_masks.reshape((-1, mh, mw))

        # Make the masks binary
        binary_masks = (post_processed_masks > 0.5).astype("uint8")

        # Make the masks channel last to scale them up
        binary_masks_t = np.transpose(binary_masks, (1, 2, 0))

        # Scale binary masks to the initial image size
        scaled_masks = cv2.resize(binary_masks_t, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Convert the masks back to channel first
        if nmsed_detections.shape[0] == 1:
            scaled_masks = np.expand_dims(scaled_masks, -1)
        scaled_masks_t = np.transpose(scaled_masks, (2, 0, 1))  # (520, 520, 3) => (3, 520, 520)

        # Normalize then scale the boxes to the initial image size
        nmsed_detections[..., [0, 2]] /= width
        nmsed_detections[..., [1, 3]] /= height
        nmsed_detections[..., [0, 2]] *= image.shape[1]
        nmsed_detections[..., [1, 3]] *= image.shape[0]

        # Bounding boxes format change: cxcywh -> xyxy
        xyxy_detections = cxcywh_to_xyxy(nmsed_detections[..., 0:6])
        colors = PaletteManager()
        custom_draw(image, xyxy_detections, scaled_masks_t, colors.get_color, prediction_result_dir, file, class_names)
    else:
        print(f"No detections found in image {file}")


def predict(cfg: Optional[DictConfig] = None) -> None:
    """
    Run inference on all the images within the test set.

    Args:
        cfg (DictConfig): The configuration file.

    Returns:
        None
    """
    n_masks = 32 
    iou_threshold = cfg.postprocessing.IoU_eval_thresh
    conf_threshold = cfg.postprocessing.confidence_thresh
    model_path = cfg.general.model_path
    test_images_dir = cfg.prediction.test_files_path
    class_names = cfg.dataset.classes_file_path
    cpp = cfg.preprocessing
    channels = 1 if cpp.color_mode == "grayscale" else 3
    aspect_ratio = cpp.resizing.aspect_ratio
    interpolation = cpp.resizing.interpolation
    scale = cpp.rescaling.scale
    offset = cpp.rescaling.offset
    prediction_result_dir = os.path.join(cfg.output_dir, 'predictions')

    if not Path(model_path).suffix == ".tflite":
        raise RuntimeError("Evaluation internal error: unsupported model type")

    print("[INFO] Making predictions using:")
    print(f"  Model: {model_path}")
    print(f"  Images directory: {test_images_dir}")

    # Load the TFLite model and allocate tensors
    net = tf.lite.Interpreter(model_path=model_path)
    net.allocate_tensors()
    input_details = net.get_input_details()[0]
    input_index_quant = input_details["index"]
    output_index_quant = net.get_output_details()
    height, width, _ = input_details['shape_signature'][1:]

    for file in os.listdir(test_images_dir):
        if file.endswith(".jpg"):
            image_path = os.path.join(test_images_dir, file)
            img = read_image(image_path, channels)
            img_process = preprocess_image(img, height, width, aspect_ratio, interpolation, scale, offset)
            img_process = preprocess_input(img_process, input_details)
            
            net.set_tensor(input_index_quant, img_process)
            net.invoke()
            
            detections = net.get_tensor(output_index_quant[0]["index"])
            masks = net.get_tensor(output_index_quant[1]["index"])
            
            postprocess(img, masks, detections, output_index_quant, conf_threshold, iou_threshold, n_masks, 
                        width, height, prediction_result_dir, file, class_names)
