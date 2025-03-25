# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/


import hydra
from omegaconf import DictConfig
import os
import sys
#SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(os.path.dirname(__file__), '../common/evaluation'))
from on_target_evaluation import gen_load_val


@hydra.main(config_path="./src/config_file_examples", config_name="evaluation_on_target_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Generates the model to be flashed on the stm32n6 device using stedgeai core,
    then loads it and validates in on the device if required.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """
    
    # Configuration information extraction
    model_path =  os.path.realpath(cfg.general.model_path)
    yaml_path = os.path.dirname(os.path.realpath(cfg.general.config_path))
    yaml_name = os.path.basename(os.path.realpath(cfg.general.config_path))
    dataset_path = os.path.realpath(cfg.general.dataset_path)
    evaluation_target = cfg.parameters.evaluation_target

    # Generates the model to be loaded on the stm32n6 device using stedgeai core,
    # then loads it and validates in on the device if required.
    gen_load_val(cfg)

    # Launches evaluation on the target through the model zoo evaluation service
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('./src')
    main_exec_command = f'python stm32ai_main.py --config-path {yaml_path} --config-name {yaml_name}'        
    os.system(f'{main_exec_command} ++operation_mode=evaluation ++general.num_threads_tflite=8 ++general.gpu_memory_limit=2 ++evaluation.target={evaluation_target.lower()} ++general.model_path={model_path} ++dataset.test_path={dataset_path} ++dataset.quantization_path="" ++dataset.training_path="" ++dataset.validation_path="" ') 

    
if __name__ == '__main__':
    main()

    
