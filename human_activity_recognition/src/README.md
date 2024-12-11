# Human Activity Recognition (HAR) STM32 model zoo

This directory contains scripts and tools for training, evaluating, and deploying HAR models using **TensorFlow** & **STM32Cube.AI**.

Remember that minimalistic yaml files are available [here](./config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!

## Training
Under [training](training/README.md), you can find a step by step guide plus the necessary scripts and tools to train and evaluate the HAR models on custom and public datasets.

## Evaluate
Under [evaluation](./evaluation/README.md), you can find a step by step guide plus the necessary scripts and tools for evaluating your model performances if datasets are provided.

## Benchmarking
Under [benchmarking](./benchmarking/README.md), you can find a step by step guide plus the necessary scripts and tools to benchmark your model using STM32Cube.AI through our STM32Cube.AI Developer Cloud Services or from the local download.


## Deployment
Under [deployment](../deployment/README.md), you can find a step by step guide plus the necessary scripts and tools to deploy your own pre-trained HAR model on your STM32 board using STM32Cube.AI.

You can also use a pretrained model from our `Human Activity Recognition STM32 model zoo`. Check out the available models in the [human_activity_recognition/pretrained_models](../pretrained_models/README.md) directory, or on the [model zoo on GitHub](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/master/human_activity_recognition/).
