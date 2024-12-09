# How to deploy a model on a B-U585I-IOT02A board?

This tutorial explains how to deploy a model from the ST public model zoo directly on your STM32 target board. In this version deployment on the B-U585I-IOT02A and [NAME OF N6 DK] is supported.

For a more detailed tutorial, look at this [tutorial](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/deployment)

## Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Deployment: 
    - To simply deploy a quantized model on the B-U585I-IOT02A board
- chain_qd: 
    - To quantize then deploy the quantized model on the B-U585I-IOT02A board

For any details regarding the parameters of the config file, you can look to the [Deployment documentation](../../../deployment/README.md)


## Deployment yaml configuration example:

Below is an example of configuration file used to deploy a pretrained and quantized yamnet from ST Model Zoo on a B-U585I-IOT02A.
You can find other example of user_config.yaml [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/src/config_file_examples)

The most important parts here are to define:
- The model path of the model to be deployed. This model must be a quantized TFLite or ONNX QDQ model.
- The operation mode to deployment
- The name of the classes
- Copy the preprocessing and feature extraction used during the training of the model (n_fft must be a power of 2)
- Select the example code deployed (sensing_free_rtos or sensing_thread_x)
- Set the target to B-U585I-IOT02A as it is the only one supported
- Select a way (or not) to handle unknown classes (see below)
- Add your local path of STM32CubeIDE


user_config.yaml
```yaml
general:
  project_name: aed_project
  model_path: ../../../model_zoo/audio_event_detection/yamnet/ST_pretrainedmodel_public_dataset/esc10/yamnet_256_64x96_tl/yamnet_256_64x96_tl_int8.tflite
  # Change this path to the model you wish to use
  logs_dir: logs
  saved_models_dir: saved_models
  global_seed: 120
  gpu_memory_limit: 5
  display_figures: True 

operation_mode: deployment

dataset:
  name: custom
  class_names: ['dog', 'chainsaw', 'crackling_fire', 'helicopter', 'rain', 'crying_baby', 'clock_tick', 'sneezing', 'rooster', 'sea_waves']
  file_extension: '.wav'

  multi_label: False 
  use_garbage_class: False 
  n_samples_per_garbage_class: 2
  expand_last_dim: True
  seed: 120 # Optional, there is a default seed
  to_cache: True
  shuffle: True

# Copy the proprocessing used in training
preprocessing:
  min_length: 1
  max_length : 10
  target_rate: 16000
  top_db: 60
  frame_length: 3200
  hop_length: 3200
  trim_last_second: False
  lengthen : 'after'

# Copy the feature extraction used in training
feature_extraction:
  patch_length: 96
  n_mels: 64
  overlap: 0.25
  n_fft: 512
  hop_length: 160
  window_length: 400
  window: hann
  center: False
  pad_mode: constant
  power: 1.0
  fmin: 125
  fmax: 7500
  norm: None
  htk : True
  to_db : False
  include_last_patch: False

tools:
  stedgeai:
    version: 10.0.0
    optimization: balanced
    on_cloud: True
    # edit paths to run
    path_to_stedgeai: C:/Users/<XXXXX>/STM32Cube/Repository/Packs/STMicroelectronics/X-CUBE-AI/<*.*.*>/Utilities/windows/stedgeai.exe
  path_to_cubeIDE: C:/ST/STM32CubeIDE_1.17.0/STM32CubeIDE/stm32cubeide.exe # Mandatory
  
deployment:
  c_project_path: ../../stm32ai_application_code/sensing_free_rtos # sensing_free_rtos or sensing_thread_x
  IDE: GCC
  verbosity: 1
  hardware_setup:
    serie: STM32U5
    board: B-U585I-IOT02A
  unknown_class_threshold: 0.5 # Threshold used for OOD detection. Mutually exclusive with use_garbage_class
                               # Set to 0 to disable. To enable, set to any float between 0 and 1.

mlflow:
  uri: ./experiments_outputs/mlruns

hydra:
  run:
    dir: ./experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
```

You can look at user_config.yaml examples for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/src/config_file_examples)

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the folder /src). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```

## Important note on "unknown" or "other" classes

A common issue in audio event detection applications is being able to reject samples which do not come from one of the classes the model is trained on. The model zoo provides several baseline options for doing this.
To help solve this issue, there are two option available:
- The first option consists of thresholding the network output probabilities at runtime. This is a naïve baseline which does not yield great results, but is a good starting point.
- The second option for OOD detection consists of adding an additional "Other" class to your model at training time, using samples from the dataset which do not belong to any of the classes specified in dataset.class_names.

**IMPORTANT NOTE:** These two methods are NOT COMPATIBLE, and cannot be used together. You must enable one or the other, or none at all. By default, in the yaml above, only the first option is in use.

You can find all the information needed [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/deployment#3)

## Local deployment:

You can use the [STM32 developer cloud](https://stedgeai-dc.st.com/home) to access the STM32Cube.AI functionalities without installing the software. This requires internet connection and making a free account. Or, alternatively, you can install [STM32Cube.AI](https://www.st.com/en/embedded-software/x-cube-ai.html) locally. In addition to this you will also need to install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html) for building the embedded project.
 
For local installation :
 
- Download and install [STM32CubeIDE](https://www.st.com/en/development-tools/stm32cubeide.html).
- If opting for using [STM32Cube.AI](https://www.st.com/en/embedded-software/x-cube-ai.html) locally, download it then extract both `'.zip'` and `'.pack'` files.
The detailed instructions on installation are available in this [wiki article](https://wiki.st.com/stm32mcu/index.php?title=AI:How_to_install_STM32_model_zoo).

## Application on the board

Once flashed the board can be connected through a serial terminal and the output of the inference can be seen in the serial terminal. 
Find more information [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/audio_event_detection/deployment#5)


