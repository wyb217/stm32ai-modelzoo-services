import cv2
import tensorflow as tf
import numpy as np

def load_quantized_model(model_path):
    # 加载量化的TFLite模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image, target_size):
    # 预处理图像：调整大小并转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = np.expand_dims(image, axis=0)  # 添加批次维度
    image = image.astype(np.float32)
    return image

def predict(interpreter, image):
    # 获取输入输出的张量
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = tuple(input_details[0]['shape'])
    scale, zero_point = input_details[0]['quantization']

    # 量化输入图像
    image = image / scale + zero_point
    image = np.clip(image, np.iinfo(input_details[0]['dtype']).min, np.iinfo(input_details[0]['dtype']).max)
    image = image.astype(input_details[0]['dtype'])

    # 设置输入张量并进行推理
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    # 获取输出
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

def display_results(image, predictions, cfg):
    # 后处理和展示结果
    boxes, scores, classes = predictions  # 假设模型输出格式为 boxes, scores, classes
    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]

    for i in range(len(boxes)):
        if scores[i] > 0.5:  # 过滤低置信度的框
            x1, y1, x2, y2 = boxes[i]
            class_name = f"Class {int(classes[i])}"
            score = scores[i]

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {score:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)  # 等待按键以关闭窗口
    cv2.destroyAllWindows()

def run_inference(model_path, image_path):
    # 加载量化模型
    interpreter = load_quantized_model(model_path)

    # 获取输入的尺寸
    input_details = interpreter.get_input_details()
    input_shape = tuple(input_details[0]['shape'][1:])
    target_size = input_shape[:2]

    # 读取输入图片
    image = cv2.imread(image_path)
    if image is None:
        print("无法加载图片")
        return

    # 预处理图像
    image_resized = preprocess_image(image, target_size)

    # 进行推理
    predictions = predict(interpreter, image_resized)

    # 显示结果
    display_results(image, predictions, None)

if __name__ == "__main__":
    model_path = r"/object_detection/src/yolov8n_256_quant_pc_uf_od_coco-person-st.tflite"  # 修改为您的模型路径
    image_path = r"D:\code\stm32ai-modelzoo-services\object_detection\datasets\img.png"  # 修改为您的输入图片路径
    run_inference(model_path, image_path)
