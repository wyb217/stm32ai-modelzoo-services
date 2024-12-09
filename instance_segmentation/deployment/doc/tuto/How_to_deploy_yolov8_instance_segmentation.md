# How to quantize, evaluate and deploy Yolo_v8 instance segmentation models for STM32N6

## Notice

Notice regarding usage of Ultralytics software [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics):

If You combine this software ("Software") with other software from STMicroelectronics ("ST Software"),  
to generate a software or software package ("Combined Software"), for instance for use in or in  
combination with STM32 products, You must comply with the license terms under which ST distributed   
such ST Software ("ST Software Terms"). Since this Software is provided to You under AGPL-3.0-only  
license terms, in most cases (such as, but not limited to, ST Software delivered under the terms of   
SLA0044, SLA0048, or SLA0078), ST Software Terms contain restrictions which will strictly forbid any   
distribution or non-internal use of the Combined Software. You are responsible for compliance with  
applicable license terms for any Software You use, and as such, You must limit your use of this  
software and any Combined Software accordingly.

## ST Ultralytics fork 

The STMicroelectronics Ultralytics fork: [https://github.com/stm32-hotspot/ultralytics/tree/main/examples/YOLOv8-STEdgeAI](https://github.com/stm32-hotspot/ultralytics/tree/main/examples/YOLOv8-STEdgeAI) provides a collection of pre-trained and quantized yolov8 models. These models are compatible with STM32 platforms, ensuring seamless integration and efficient performance for edge computing applications.
- Offers a set of pre-trained yolov8 models compatible with STM32 platforms and stm32ai-modelzoo.
- Offers a quantization friendly yolov8 pose estimation model.
These models are ready to be deployed and you can go directly to the deployment section.
The other sections below explain how to start from a model trained with Ultralytics scripts and not quantized.

## Training a model with Ultralytics scripts

Train the `Yolov8n-seg` model as usual using Ultralytics scripts or start from the pre-trained Yolov8n-seg Pytorch model.
Please refer to [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).


## Exporting a quantized model with Ultralytics scripts

To export the model as int8 tflite, let's take the default example of a model trained on COCO dataset using Ultralytics CLI:

```
	yolo export model=yolov8n-seg.pt format=tflite imgsz=256 int8=True
```

Where yolov8n-seg.pt is the trained weights, the output format is specified as tflite, int8=True means the model will be quantized using 8-bits signed for the weights and the activations.  
In this case the default COCO 2017 dataset is used by Ultralytics to pre-train the models.  
If no data for calibration is specified, a very small subset of images of COCO 2017 will be downloaded and the 4 images of the validation set will be used to calibrate.
From our experiment, the quantization is nevertheless efficient even more than using the full validation dataset of COCO (that you can test by using "data=coco.yaml").


By default the exported models are:
1. A tensorflow float saved model, the saved model generated differs from exporting directly to saved model as the output is normalized to allow quantization: yolov8n-seg_saved_model directory.
2. An onnx float model: yolov8n-seg.onnx.
3. A quantized model with input / output in integer int8 format: yolov8n-seg_saved_model/yolov8n-seg_integer_quant.tflite.
4. A quantized model with input / output in float format: yolov8n-seg_saved_model/yolov8n-seg_full_integer_quant.tflite.

By default, the model is quantized per-tensor. 
It is recommended to use per-channel quantization to better maintain the accuracy, so we recommend to use directly tensorflow lite converter to do the quantization.
> To do so in Ultralytics git repository, open the file ultralytics/
> and modify the following line of the onn2tf.convert callback:
> quant_type="per-tensor",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)
> to
> quant_type="per-channel",  # "per-tensor" (faster) or "per-channel" (slower but more accurate)

For deployment the model shall be quantized with input as int8 and output as int8: yolov8n-seg_saved_model/yolov8n-seg_integer_quant.tflite


## Evaluation of the quantized model with Ultralytics scripts

Use the model with int8 input and output, then use the CLI to evaluate the model onn the COCO validation set:

```
	yolo segment val model=./quantized_models/yolov8n-seg_integer_quant.tflite data=coco.yaml imgsz=256
```

## Deployment of the quantized model on the STM32N6

The models with int8 input and int8 output can be deployed using the STM32 model zoo deployment service:
[https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/instance_segmentation/deployment](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/instance_segmentation/deployment)

