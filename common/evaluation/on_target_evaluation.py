# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2022 STMicroelectronics.
#  * All rights reserved.
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from omegaconf import DictConfig
import pandas as pd
import math
import os
import pathlib
import subprocess
import sys
import shlex


def gen_load_val(cfg: DictConfig) -> None:
    """
    Generates the model to be loaded on the stm32n6 device using stedgeai core,
    then loads it and validates in on the device if required.

    Args:
        cfg: Configuration dictionary.

    Returns:
        None
    """
    
    # Configuration information extraction
    model_path =  os.path.realpath(cfg.general.model_path)
    profile = cfg.parameters.profile
    input_type = cfg.parameters.input_type
    output_type = cfg.parameters.output_type
    input_chpos = cfg.parameters.input_chpos
    output_chpos = cfg.parameters.output_chpos
    stedgeai_path = os.path.realpath(cfg.tools.stedgeai.path_to_stedgeai)
    loader_path = os.path.realpath(cfg.tools.stedgeai.path_to_loader)
    stedgeai_exe = f'{stedgeai_path}'
    loader = f'{loader_path}'
    
    # Then Depending on the targetted evaluation on device, execute on the host (TFLite interpreter), on the host emulating STM32 MCU device or on the STM32N6 device
    evaluation_target = cfg.parameters.evaluation_target
    if evaluation_target.lower()=="stedgeai_n6":
        os.chdir(os.path.dirname(loader))
        print("stedgeai_n6 validation")
        # Generates of model to be flashed on the stm32n6 device
        exec_string = f'{stedgeai_exe} generate --c-api st-ai --model {model_path} --target stm32n6 ' +\
                      f'--st-neural-art {profile}@user_neuralart.json ' +\
                      f'--input-data-type {input_type} ' +\
                      f'--output-data-type {output_type} ' +\
                      f'--inputs-ch-position {input_chpos} ' +\
                      f'--outputs-ch-position {output_chpos} '
        args = shlex.split(exec_string, posix="win" not in sys.platform)
        subprocess.run(args, check=True)
    
        # Loads on Device
        exec_string = f'python {loader} ' +\
                      f'--n6-loader-config ./config_n6l.json '
        os.system(exec_string) 

        # Validates on Device
        exec_string = f'{stedgeai_exe} validate --c-api st-ai --model {model_path} --target stm32n6 ' +\
                      f'--mode target -d serial:921600 -b 1 ' +\
                      f'--input-data-type {input_type} ' +\
                      f'--output-data-type {output_type} ' +\
                      f'--inputs-ch-position {input_chpos} ' +\
                      f'--outputs-ch-position {output_chpos} '
        args = shlex.split(exec_string, posix="win" not in sys.platform)
        subprocess.run(args, check=True)
        
    elif evaluation_target.lower()=="stedgeai_host":
        print("stedgeai_host validation")
        # Generates dll for emulated embedded C code execution
        os.chdir('./src')
        exec_string = f'{stedgeai_exe} generate --c-api st-ai --model {model_path} ' +\
                      f'--target stm32 ' +\
                      f'--dll ' +\
                      f'--input-data-type {input_type} ' +\
                      f'--output-data-type {output_type} ' +\
                      f'--inputs-ch-position {input_chpos} ' +\
                      f'--outputs-ch-position {output_chpos} '
        args = shlex.split(exec_string, posix="win" not in sys.platform)
        subprocess.run(args, check=True)
    
    elif evaluation_target.lower()=="host":
        print("host validation")

    else:
        raise ValueError(f"\nUnsupported evaluation target.")

    