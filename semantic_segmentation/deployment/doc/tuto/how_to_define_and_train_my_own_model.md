# How can I define and train my own model with ST Model Zoo?

With ST Model Zoo, you can easily define and train your own TensorFlow neural network model.

## Define my model

First, create your own model in /src/models/custom_model.py for it to be automatically used with the model zoo training script.
Open the python file and copy your model topology inside, here is the default example model:

```python
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers


def get_custom_model(num_classes: int = None, input_shape: Tuple[int, int, int] = None,
                     dropout: Optional[float] = None) -> tf.keras.Model:
    """
    Creates a custom segmentation model

    Args:
        num_classes (int): Number of classes in the segmentation task.
        input_shape (Tuple[int, int, int]): Shape of the input image.
        dropout (Optional[float]): Dropout rate to be applied to the model.

    Returns:
        keras.Model: Custom segmentation model.
    """
    # Define the input layer
    inputs = tf.keras.Input(shape=input_shape)

    # Define the feature extraction layers, plays the role of a backbone. This is only given as example.
    x = layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # insert a couple of layers playing the role of an Head. This is only given as example.
    x = layers.Conv2D(256, 1, strides=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # to finish, add the segmentation layers
    x = layers.Dropout(rate=dropout)(x)
    # Add a segmentation layer with the number of classes
    x = layers.Conv2D(num_classes, 1, strides=1, kernel_initializer='he_normal')(x)
    # Up sample the features to the original size of the input, stride of feature extractor being 16 in this example.
    x = layers.UpSampling2D(size=(16, 16), interpolation='nearest')(x)

    # Construct the final model
    model = tf.keras.Model(inputs=inputs, outputs=x, name="custom_model")
    return model
```

The model must be created inside the function get_custom_model. The input size and number of classes are then define in the user_config.yaml. See below.


## Training my model

To train the model, we then edit the user_config.yaml and run the training using the python script stm32ai_main.py.

### Dataset
For this example, we used the [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) dataset.

Copy the datasets to /sementic_segmentation/datasets/ and you need to have your file structure like this:

```yaml
dataset_root_directory/
   Images/
      image_1.jpg
      image_2.jpg
      ...
   Segmentation_masks/
      mask_1.png
      mask_2.png
      ...
   Image_sets/
      train.txt
      val.txt
```
A directory contains all the images used for training, validation, and testing, and another one holds the segmentation masks corresponding to the images and the last one is for text files like train.txt and val.txt which list the filenames of images that are included in the training and validation sets, respectively.

More details [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/blob/main/semantic_segmentation/src/training/README.md#1)

### Operation modes:

Depending on what you want to do, you can use the operation modes below:
- Training:
    - To simply train the model and get as output the trained tensorflow model (.h5).
- Chain_tqe:
    - To train, quantize and evaluate the model in one go. You get as ouput both the train and quantized trained models (.h5 and .tflite)
- Chain_tbqeb:
    - To train, benchmark, quantize, evaluate and benchmark again the model in one go.

For any details regarding the parameters of the config file, you can look here:

- [Training documentation](../../../src/training/README.md)
- [Quantization documentation](../../../src/quantization/README.md)
- [Benchmark documentation](../../../src/benchmarking/README.md)
- [Evaluation documentation](../../../src/evaluation/README.md)


### Benchmarking Configuration example:

The most important parts here are to define:
- The operation mode to training
- The data paths
- Define which and how many classes you segment
- The model name to custom for model zoo to load the model in custom_model.py
- The input_shape and other training parameters

```yaml
# user_config.yaml

general:
  project_name: segmentation
  saved_models_dir: saved_models
  gpu_memory_limit: 12
  global_seed: 127
  display_figures: False

operation_mode: training

dataset:
  name: pascal_voc
  # class to detect/segment
  class_names: ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike",
                "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]
  # path for data to be use for the training of the model
  training_path: ../datasets/VOC2012_train_val/JPEGImages
  training_masks_path: ../datasets/VOC2012_train_val/SegmentationClassAug
  training_files_path: ../datasets/VOC2012_train_val/ImageSets/Segmentation/trainaug.txt
  # path for data to be use for the validation of the model
  validation_path: ../datasets/VOC2012_train_val/JPEGImages
  validation_masks_path: ../datasets/VOC2012_train_val/SegmentationClassAug
  validation_files_path: ../datasets/VOC2012_train_val/ImageSets/Segmentation/val.txt
  validation_split: 
  
# preprocessing to rescale and resize the data to the model input size define below
preprocessing:
  rescaling: {scale: 1/127.5, offset: -1}
  resizing:
    aspect_ratio: fit
    interpolation: bilinear 
  color_mode: rgb

# Optional
data_augmentation:   
  random_contrast:
    factor: 0.4
    change_rate: 1.0
  random_gaussian_noise:
    stddev: (0.0001, 0.005)
  random_posterize:
    bits: (4, 8)
    change_rate: 0.025
  random_brightness:
    factor: 0.05
    change_rate: 1.0

training:
  model:
    name: custom # put custom for the script to understand that you want to use your own model
    input_shape: (128, 128, 3) # your input size
    # Any other argument you added to the get_custom_model() function

  # all the parameters below are standard in machine learning, you can look for them in google
  # they mostly depends on the topology of your model and will need a lot of testing
  dropout: 0.6
  batch_size: 16
  epochs: 1
  optimizer:
    Adam:
      learning_rate: 0.005
  callbacks:          
    ReduceLROnPlateau:
      monitor: val_accuracy
      mode: max
      factor: 0.5
      patience: 40
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_accuracy
      mode: max
      restore_best_weights: true
      patience: 60


mlflow:
  uri: ./experiments_outputs/mlruns

hydra:
  run:
    dir: ./experiments_outputs/${now:%Y_%m_%d_%H_%M_%S}
  
```

You can also find example of user_config.yaml for any operation mode [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services/tree/main/semantic_segmentation/src/config_file_examples)

## Run the script:

Edit the user_config.yaml then open a terminal (make sure to be in the folder /src). Finally, run the command:

```powershell
python stm32ai_main.py
```
You can also use any .yaml file using command below:
```powershell
python stm32ai_main.py --config-path=path_to_the_folder_of_the_yaml --config-name=name_of_your_yaml_file
```