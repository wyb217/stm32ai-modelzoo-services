# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
''' Copied from common_deploy, trying to avoid Tensorflow dependencies yet again'''
from typing import Dict
import os
import shutil
from pathlib import Path
import sys
sys.path.append('../../../common/')
import stm32ai_local as stmaic
from utils import cloud_connect

def stm32ai_deploy_stm32n6(target: bool = False,
                   stlink_serial_number: str = None,
                   stm32ai_version: str = None,
                   c_project_path: str = None,
                   output_dir: str = None,
                   stm32ai_output: str = None,
                   optimization: str = None,
                   path_to_stm32ai: str = None,
                   path_to_cube_ide: str = None,
                   additional_files: list = None,
                   stmaic_conf_filename: str = 'stmaic_c_project.conf',
                   verbosity: int = None,
                   debug: bool = False,
                   model_path: str = None,
                   get_model_name_output: str = None,
                   stm32ai_ide: str = None,
                   stm32ai_serie: str = None,
                   credentials: list[str] = None,
                   on_cloud: bool =False,
                   check_large_model:bool = False,
                   build_conf: str = None,
                   cfg = None,
                   custom_objects: Dict = None,
                   input_data_type: str = '',
                   output_data_type: str = '',
                   inputs_ch_position: str = '',
                   outputs_ch_position: str = '') -> None:
    """
    Deploy an STM32 AI model to a target device.

    Args:
        target (bool): Whether to generate the STM32Cube.AI library and header files on the target device. Defaults to False.
        c_project_path (str): Path to the STM32CubeIDE C project.
        verbosity (int, optional): Level of verbosity for the STM32Cube.AI driver. Defaults to None.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.
        model_path (str, optional): Path to the AI model file. Defaults to None.
        on_cloud(bool): whether to deploy using the cloud. Defaults to False
        config(list):

    Returns:
        split_weights (bool): return true if the weights has been splitted; False otherwise
    """
    def stmaic_local_call(session):
        """
        Compile the AI model using the STM32Cube.AI compiler.

        Args:
            session (stmaic.STMAiSession): The STM32Cube.AI session object.

        Returns:
            None
        """
        # Add environment variables
        os.environ["STM32_AI_EXE"] = path_to_stm32ai
        # Set the tools
        tools = stmaic.STMAiTools()
        session.set_tools(tools)
        print("[INFO] : Offline CubeAI used; Selected tools: ", tools, flush=True)

        # Clean up the STM32Cube.AI output directory
        shutil.rmtree(stm32ai_output, ignore_errors=True)

        # Set the compiler options
        neural_art_path = session._board_config.config.profile + "@" + session._board_config.config.neuralart_user_path
        opt = stmaic.STMAiCompileOptions(st_neural_art=neural_art_path, input_data_type=input_data_type, inputs_ch_position=inputs_ch_position,
                                         output_data_type = output_data_type, outputs_ch_position = outputs_ch_position)

        # 2 - set the board configuration
        board_conf = os.path.join(c_project_path, stmaic_conf_filename)
        board = stmaic.STMAiBoardConfig(board_conf, build_conf)
        session.set_board(board)

        # Compile the AI model
        stmaic.compile(session=session, options=opt, target=session._board_config)

     # Add environment variables
    os.environ["STM32_CUBE_IDE_EXE"] = path_to_cube_ide

    # Set the level of verbosity for the STM32Cube.AI driver
    if debug:
        stmaic.set_log_level('debug')
    elif verbosity is not None:
        stmaic.set_log_level('info')

    # 1 - create a session
    session = stmaic.load(model_path, workspace_dir=output_dir)

    # 2 - set the board configuration
    board_conf = os.path.join(c_project_path, stmaic_conf_filename)
    board = stmaic.STMAiBoardConfig(board_conf, build_conf)
    session.set_board(board)
    print("[INFO] : Selected board : ", board, flush=True)
    # 3 - compile the model
    user_files = []
    print("[INFO] : Compiling the model and generating optimized C code + Lib/Inc files: ", model_path, flush=True)

    if on_cloud:
        raise ValueError("Dev cloud is not supported to deploy on N6")
    else:
        stmaic_local_call(session)

    print("[INFO] : Optimized C code + Lib/Inc files generation done.")

    # 4 - build and flash the STM32 c-project
    print("[INFO] : Building the STM32 c-project..", flush=True)

    user_files.extend([os.path.join(output_dir, "C_header/app_config.h")])
    user_files.extend([os.path.join(output_dir, "C_header/ai_model_config.h")])
    if additional_files:
        for f in additional_files:
            user_files.extend([os.path.join(output_dir, f)])

    stmaic.build(session, user_files=user_files, serial_number=stlink_serial_number)
