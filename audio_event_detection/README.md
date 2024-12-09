# Audio event detection (AED) STM32 model zoo

## Directory components:
* [datasets](datasets/README.md) placeholder for the audio event detection datasets.
* [deployment](deployment/README.md) contains the necessary files to deploy models on an STM32 board.
* [pretrained_models](pretrained_models/README.md) points to a collection of optimized pretrained models on different audio datasets and provides models performances.
* [src](src/README.md) contains tools to train, evaluate, benchmark, and quantize your model on your STM32 target.

## Tutorials and documentation: 
* [Complete AED model zoo and configuration file documentation](src/README.md)
* [A short tutorial on training a model using the model zoo](src/training/README.md)
* [A short tutorial on quantizing a model using the model zoo](src/quantization/README.md)
* [A short tutorial on deploying a model on an STM32 board](deployment/README.md)

All .yaml configuration examples are located in [config_file_examples](./src/config_file_examples/) folder.

The different values of the `operation_mode` attribute and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations |
|:-------------------------|:-----------|
| `training`               | Train a model  |
| `evaluation`             | Evaluate the accuracy of a float or quantized model on a test or validation dataset|
| `quantization`           | Quantize a float model |
| `prediction`             | Predict the classes some audio events belong to using a float or quantized model |
| `benchmarking`           | Benchmark a float or quantized model on an STM32 board |
| `deployment`             | Deploy a model on an STM32 board |
| `chain_tbqeb`             | Sequentially: training, benchmarking, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`              | Sequentially: training, quantization of trained model, evaluation of quantized model |
| `chain_eqe`              | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model |
| `chain_qb`               | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`             | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`               | Sequentially: quantization of a float model, deployment of quantized model |


## You don't know where to start? You feel lost?
Don't forget to follow our tuto below for a quick ramp up : 
* [How to define and train my own model?](../audio_event_detection/deployment/doc/tuto/how_to_define_and_train_my_own_model.md)
* [How to fine tune a model on my own dataset?](../audio_event_detection/deployment/doc/tuto/how_to_finetune_a_model_zoo_model_on_my_own_dataset.md)
* [How can I evaluate my model before and after quantization?](../audio_event_detection/deployment/doc/tuto/how_to_compare_the_accuracy_after_quantization_of_my_model.md)
* [How can I quickly check the performance of my model using the dev cloud?](../audio_event_detection/deployment/doc/tuto/how_to_quickly_benchmark_the_performances_of_a_model.md)
* [How can I deploy my model?](../audio_event_detection/deployment/doc/tuto/how_to_deploy_a_model_on_a_target.md)

Remember that minimalistic yaml files are available [here](./src/config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!
