# How can I finetune a ST Model Zoo model on my own dataset?

With ST Model Zoo, you can easily pick an already pretrained available model and finetune them on your own dataset.

## Pick a pretrained model

A choice of model architectures pretrained on multiple dataset can be found [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/pose_estimation/pretrained_models).

## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model on my data and get as output the trained tensorflow model (.h5).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.h5 and .tflite)
- Chain_tqeb:
    - To train, quantize, evaluate and benchmark the quantized model in one go.

For any details regarding the parameters of the config file, you can look here:
- [Training documentation](../../../src/training/README.md)
- [Quantization documentation](../../../src/quantization/README.md)
- [Benchmark documentation](../../../src/benchmarking/README.md)
- [Evaluation documentation](../../../src/evaluation/README.md)


## Finetune the model on my dataset

To retrain the model we edit the user_config.yaml and the stm32ai_main.py python script (both found in /src).

In this example, we retrain our ST MoveNet Lightning heatmap model with an input size of (192x192x3) pretrained on a large public dataset imagenet, with our data.


Here, we used the COCO2017 pose estimation dataset. 

You can use any dataset of the YOLO Darknet format. You can take a look at this [tutorial](./how_to_use_my_own_dataset.md) which explain how to convert a COCO dataset using our script.

The most important parts here are to define:
- The operation mode to training
- The data paths
- Define the number of keypoints of the pose
- Choose a model, its pretrained weights and input size
- The other training parameters

```yaml
# user_config.yaml

general:
  project_name: COCO_2017_pose_Demo
  logs_dir: logs
  saved_models_dir: saved_models
  model_path:
  model_type: heatmaps_spe
  num_threads_tflite: 8
  gpu_memory_limit: 8
  global_seed: 123

operation_mode: training

dataset:
  name: COCO2017_pose
  keypoints: 17
  training_path: ../datasets/coco_train_single_pose
  # validation_path: ../datasets/coco_val_single_pose
  validation_split: 0.1
  test_path: ../datasets/coco_val_single_pose
  # quantization_path: ../datasets/coco_train_single_pose
  quantization_split: 0.3

preprocessing:
  rescaling: { scale: 1/127.5, offset: -1 }
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: rgb

# Optional
data_augmentation:
  random_periodic_resizing:
    image_sizes: [[192,192],[224,224],[256,256]]
  random_contrast:
    factor: 0.4
  random_brightness:
    factor: 0.3
  random_flip:
    mode: horizontal
  random_rotation:
    factor: (-0.1,0.1) # -+0.1 = -+36 degree angle

training:
  model:
    name: st_movenet_lightning_heatmaps
    alpha: 1.0
    input_shape: (192, 192, 3)
    pretrained_weights: imagenet
  resume_training_from: # experiments_outputs/2024_11_06_16_44_31/
  frozen_layers: # (0:154)
  batch_size: 64
  epochs: 1000
  optimizer:
    Adam:
      learning_rate: 0.01
  callbacks:
    ReduceLROnPlateau:
      monitor: val_oks
      mode: max
      factor: 0.25
      min_delta: 0.0001
      patience: 5
    ModelCheckpoint:
      monitor: val_oks
      mode: max
    EarlyStopping:
      monitor: val_oks
      mode: max
      min_delta: 0.0001
      patience: 10

mlflow:
  uri: ./experiments_outputs/mlruns

hydra:
  run:
    dir: ./experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

You can also find examples of user_config.yaml for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/pose_estimation/src/config_file_examples)

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the folder /src). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```