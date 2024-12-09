# Speech enhancement STM32 model zoo

## Directory Components:
* [src/deployment](src/deployment/README.md) contains the necessary files and scripts to deploy a speech enhancement model on STM32N6.
* [src](src/README.md) contains code to train, evaluate, benchmark and quantize speech enhancement models


## Quick & easy examples:
The `operation_mode` top-level parameter in the [configuration file](./src/user_config.yaml) specifies the operations or the service you want to execute. This may be a single operation or a set of chained operations.

You can refer to the README links below that provide typical examples of operation modes, and tutorials on specific services:
   - [training](./src/trainers/README.md)
   - [quantization](./src/quantization/README.md)
   - [evaluation](./src/evaluators/README.md)
   - [deployment](./src/deployment/README.md)

To help you get started, configuration file examples are located in the [config_file_examples](./src/config_file_examples/) folder.

The different values of the `operation_mode` parameter and the corresponding operations are described in the table below. In the names of the chain modes, 't' stands for training, 'e' for evaluation, 'q' for quantization, 'b' for benchmark and 'd' for deployment on an STM32 board.

| operation_mode attribute | Operations |
|:---------------------------|:-----------|
| `training`| Train a speech enhancement model, using an architecture provided in the zoo, or your own custom architecture.|
| `evaluation` | Evaluate a float or a quantized speech enhancement model|
| `quantization` | Quantize a speech enhancement model |
| `benchmarking` | Benchmark a float or quantized model on an STM32 board |
| `deployment`   | Deploy a speech enhancement model on an STM32 board |
| `chain_tqeb`  | Sequentially: training, quantization of trained model, evaluation of quantized model, benchmarking of quantized model |
| `chain_tqe`    | Sequentially: training, quantization of trained model, evaluation of quantized model |
| `chain_eqe`    | Sequentially: evaluation of a float model,  quantization, evaluation of the quantized model |
| `chain_qb`     | Sequentially: quantization of a float model, benchmarking of quantized model |
| `chain_eqeb`   | Sequentially: evaluation of a float model,  quantization, evaluation of quantized model, benchmarking of quantized model |
| `chain_qd`     | Sequentially: quantization of a float model, deployment of quantized model |

## You don't know where to start? You feel lost?
Remember that minimalistic yaml files are available [here](./src/config_file_examples/) to play with specific services, and that all pre-trained models in the [STM32 model zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo/) are provided with their configuration .yaml file used to generate them. These are very good starting points to start playing with!
