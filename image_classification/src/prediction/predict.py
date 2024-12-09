# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import cv2
import sys
from pathlib import Path
from omegaconf import DictConfig
from tabulate import tabulate
import numpy as np
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from models_utils import get_model_name_and_its_input_shape, ai_runner_interp, ai_interp_input_quant, ai_interp_outputs_dequant
from models_mgt import ai_runner_invoke
from preprocess import preprocess_input
from onnx_evaluation import predict_onnx
import onnxruntime


def load_test_data(directory: str):
    """
    Parse the training data and return a list of paths to annotation files.
    
    Args:
    - directory: A string representing the path to test set directory.
    
    Returns:
    - A list of strings representing the paths to test images.
    """
    annotation_lines = []
    path = directory+'/'
    for file in os.listdir(path):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg"):
            new_path = path+file
            annotation_lines.append(new_path)
    return annotation_lines

def predict(cfg: DictConfig = None) -> None:
    """
    Predicts a class for all the images that are inside a given directory.
    The model used for the predictions can be either a .h5 or .tflite file.

    Args:
        cfg (dict): A dictionary containing the entire configuration file.

    Returns:
        None
    
    Errors:
        The directory containing the images cannot be found.
        The directory does not contain any file.
        An image file can't be loaded.
    """
    
    model_path = cfg.general.model_path
    class_names = cfg.dataset.class_names
    test_images_dir = cfg.prediction.test_files_path
    cpp = cfg.preprocessing
    if cfg.prediction and cfg.prediction.target:
        target = cfg.prediction.target
    else:
        target = "host"
    name_model = os.path.basename(model_path)

    _, model_input_shape = get_model_name_and_its_input_shape(model_path)
    
    print("[INFO] : Making predictions using:")
    print("  model:", model_path)
    print("  images directory:", test_images_dir)

    channels = 1 if cpp.color_mode == "grayscale" else 3
    results_table = []
    file_extension = Path(model_path).suffix

    if test_images_dir:
        image_filenames =  load_test_data(test_images_dir)
    else:
        print("no test set found")

    if file_extension == ".h5":
        # Load the .h5 model
        model = tf.keras.models.load_model(model_path)
    elif file_extension == ".tflite":
        # Load the Tflite model and allocate tensors
        interpreter_quant = tf.lite.Interpreter(model_path=model_path)
        interpreter_quant.allocate_tensors()
        input_details = interpreter_quant.get_input_details()[0]
        input_index_quant = input_details["index"]
        output_index_quant = interpreter_quant.get_output_details()[0]["index"]
        ai_runner_interpreter = ai_runner_interp(target,name_model)
    elif file_extension == ".onnx":
        sess = onnxruntime.InferenceSession(model_path)
        ai_runner_interpreter = ai_runner_interp(target,name_model)

    prediction_result_dir = f'{cfg.output_dir}/predictions/'
    os.makedirs(prediction_result_dir, exist_ok=True)

    for i in range(len(image_filenames)):
        if image_filenames[i].endswith(".jpg") or image_filenames[i].endswith(".png") or image_filenames[i].endswith(".jpeg"):

            print('Inference on image : ',image_filenames[i])
            im_path = os.path.join(test_images_dir, image_filenames[i])

            # Load the image with Tensorflow for the model inference
            try:
                data = tf.io.read_file(im_path)
                img = tf.image.decode_image(data, channels=channels)
            except:
                raise ValueError(f"\nUnable to load image file {im_path}\n"
                                 "Supported image file formats are BMP, GIF, JPEG and PNG.")
            # Resize the image            
            width, height = model_input_shape[1:] if Path(model_path).suffix == '.onnx' else model_input_shape[0:2]
            if cpp.resizing.aspect_ratio == "fit":
                img = tf.image.resize(img, [height, width], method=cpp.resizing.interpolation, preserve_aspect_ratio=False)
            else:
                img = tf.image.resize_with_crop_or_pad(img, height, width)
            # Rescale the image
            img = cpp.rescaling.scale * tf.cast(img, tf.float32) + cpp.rescaling.offset

            # Load the image with OpenCV to print it on screen
            image = cv2.imread(im_path)
            if len(image.shape) != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            thick = int(0.6 * (height + width) / 600)
            img_name = os.path.splitext(image_filenames[i])[0]

            if file_extension == ".h5":
                img = tf.expand_dims(img, 0)
                scores = model.predict(img)
            elif file_extension == ".tflite":
                image_processed = preprocess_input(img, input_details)
                if target == 'host':
                    interpreter_quant.set_tensor(input_index_quant, image_processed)
                    interpreter_quant.invoke()
                    scores = interpreter_quant.get_tensor(output_index_quant)
                elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                    imagee = ai_interp_input_quant(ai_runner_interpreter,img[None].numpy(),cfg.preprocessing.rescaling.scale, cfg.preprocessing.rescaling.offset,'.tflite')
                    scores = ai_runner_invoke(imagee,ai_runner_interpreter)
                    scores = ai_interp_outputs_dequant(ai_runner_interpreter,[scores])[0]
            elif file_extension == ".onnx":
                image_processed = np.expand_dims(img, 0)
                image_processed = np.transpose(image_processed,[0,3,1,2])
                if target == 'host':
                    scores = predict_onnx(sess, image_processed)
                elif target == 'stedgeai_host' or target == 'stedgeai_n6':
                    imagee = ai_interp_input_quant(ai_runner_interpreter,image_processed,cfg.preprocessing.rescaling.scale, cfg.preprocessing.rescaling.offset,'.onnx')
                    scores = ai_runner_invoke(imagee,ai_runner_interpreter)
                    scores = ai_interp_outputs_dequant(ai_runner_interpreter,[scores])[0]
            else:
                raise TypeError(f"Unknown or unsupported model type. Received path {model_path}")

            # Find the label with the highest score
            scores = np.squeeze(scores)
            if scores.shape == ():
                scores = [scores]
            max_score_index = np.argmax(scores)
            prediction_score = 100 * scores[max_score_index]
            predicted_label = class_names[max_score_index]

            # Add result to the table
            results_table.append([predicted_label, "{:.1f}".format(prediction_score), image_filenames[i]])
            pred_text = str(predicted_label) + ": " +"{:.1f}".format(prediction_score)+ "%"
            cv2.rectangle(image, pt1=(int(0.2*width//2) - int(0.037*width), int(0.2*height//2) - int(2*0.037*height)),pt2=(int(0.2*width//2) + int(len(pred_text)*0.037*width), int(0.2*height//2) + int(0.5*0.037*height)),color=[0, 0, 0], thickness=-1)
            cv2.putText(image, pred_text, (int(0.2*width//2),int(0.2*height//2)), cv2.FONT_HERSHEY_COMPLEX, width/500, (255, 255, 255), thick, lineType=cv2.LINE_AA)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # writing prediction result to the output dir
            pred_res_filename = f'{prediction_result_dir}/{os.path.basename(img_name)}.png'
            cv2.imwrite(pred_res_filename,image)
            if cfg.general.display_figures:
                cv2.imshow('image',image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()