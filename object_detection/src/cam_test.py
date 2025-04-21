import sys
import cv2
import os

import time
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from clearml import Task
from clearml.backend_config.defs import get_active_config_file

from common.utils.cfg_utils import get_random_seed
from common.utils.logs_utils import mlflow_ini
from object_detection.src.postprocessing.postprocess import get_nmsed_detections
from object_detection.src.utils.bounding_boxes_utils import bbox_normalized_to_abs_coords, plot_bounding_boxes
from object_detection.src.utils.parse_config import get_config


# Helper functions
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


def preprocess_image_for_quantized_model(frame, input_shape):
    """Process image for quantized model input"""
    # Convert the frame to RGB and resize it to match model input shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    frame = frame.astype(np.float32)  # 转换为 float32 类型

    # 归一化：将像素值范围从 [0, 255] 映射到 [0, 1]
    frame /= 255.0

    return frame


def predict_quantized_model(cfg, model_path):
    """
    使用摄像头进行实时检测
    """
    if cfg['prediction'] and cfg['prediction']['target']:
        target = cfg['prediction']['target']
    else:
        target = "host"

    name_model = Path(model_path).name
    print("加载 TFlite 模型文件:", model_path)

    # 加载 TFLite 模型
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()
    batch_size=1
    # 获取输入形状
    input_shape = tuple(input_details['shape'][1:])
    image_size = input_shape[:2]

    # 初始化摄像头
    cap = cv2.VideoCapture(0)  # 0 是默认摄像头
    print(cap.isOpened())
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理图像
        frame = preprocess_image_for_quantized_model(frame, image_size)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #量化输入数据
        scale = input_details['quantization'][0]
        zero_points = input_details['quantization'][1]
        frame += zero_points
        frame /= scale
        frame=np.clip(frame, 0, 255).astype(np.uint8)
        tensor_shape = (batch_size,) + input_shape  # 批量大小 + 图像尺寸
        input_index = input_details['index']
        interpreter.resize_tensor_input(input_index, tensor_shape)


        input_dtype = input_details['dtype']
        frame = tf.cast(frame, input_dtype)
        frame = tf.clip_by_value(frame, np.iinfo(input_dtype).min, np.iinfo(input_dtype).max)
        # 确保 frame 是一个 4D 张量，增加批次维度
        frame = np.expand_dims(frame, axis=0)  # 将 (height, width, channels) 转换为 (1, height, width, channels)

        interpreter.set_tensor(input_index, frame)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        boxes, scores, classes = get_nmsed_detections(cfg, predictions, image_size)

        boxes = boxes[0]

        scores = scores[0]
        classes = classes[0]
        # # except Exception as e:
        # #     print(f"后处理时出错: {e}")
        # #     continue
        # #
        # # 将归一化的边界框转换为绝对坐标
        boxes = bbox_normalized_to_abs_coords(boxes, image_size)
        #
        # # 将边界框、类别和分数转换为 NumPy 数组
        boxes = boxes.numpy() if isinstance(boxes, tf.Tensor) else boxes
        scores = scores.numpy() if isinstance(scores, tf.Tensor) else scores
        classes = classes.numpy() if isinstance(classes, tf.Tensor) else classes
        #
        # # 确保边界框坐标是整数
        boxes = boxes.astype(np.int32)
        classes = classes.astype(np.int32)
        scores = scores.astype(np.float32)
        #
        # # 显示图像和边界框
        frame = np.array(frame[0], dtype=np.uint8)
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            class_name = cfg.dataset.class_names[classes[i]]
            score = scores[i]
        #
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #
            # 绘制类别名称和分数
            label = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Camera Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_camera(cfg):
    model_path = cfg.general.model_path
    predict_quantized_model(cfg, model_path)


@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    """
    Main entry point of the script.
    """
    # Configure the GPU (the 'general' section may be missing)
    if "general" in cfg and cfg.general:
        # Set upper limit on usable GPU memory
        if "gpu_memory_limit" in cfg.general and cfg.general.gpu_memory_limit:
            set_gpu_memory_limit(cfg.general.gpu_memory_limit)
        else:
            print("[WARNING] The usable GPU memory is unlimited.\n"
                  "Please consider setting the 'gpu_memory_limit' attribute "
                  "in the 'general' section of your configuration file.")

    # Parse the configuration file
    cfg = get_config(cfg)
    cfg.output_dir = HydraConfig.get().runtime.output_dir
    mlflow_ini(cfg)

    # Checks if there's a valid ClearML configuration file
    print(f"[INFO] : ClearML config check")
    if get_active_config_file() is not None:
        print(f"[INFO] : ClearML initialization and configuration")
        # ClearML - Initializing ClearML's Task object.
        task = Task.init(project_name=cfg.general.project_name,
                         task_name='od_modelzoo_task')
        task.connect_configuration(name=cfg.operation_mode, configuration=cfg)

    # Seed global seed for random generators
    seed = get_random_seed(cfg)
    print(f'[INFO] : The random seed for this simulation is {seed}')
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
    predict_camera(cfg)


if __name__ == "__main__":
    main()