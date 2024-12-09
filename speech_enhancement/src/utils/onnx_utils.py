# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

import os
import onnx

'''Function taken from common/evaluation/onnx_utils, to avoid adding a Tensorflow dependency.'''
def model_is_quantized(onnx_model_path:str) -> bool:
    """
    Check if an ONNX model is quantized.

    This function iterates through all the initializers (weights) in the provided
    ONNX model to determine if any of the weights are stored as quantized data types.
    The presence of quantized data types (UINT8, INT8, INT32) among the weights
    indicates that the model is quantized. If only floating-point data types (FLOAT,
    DOUBLE) are found, the model is considered not quantized.

    Args:
    - onnx_model_path (str): The ONNX model path.

    Returns:
    - bool: True if the model is quantized, False otherwise.
    """
    if not os.path.isfile(onnx_model_path):
        raise FileNotFoundError('File does not exist!\nCheck the input onnx model path!')
    onnx_model = onnx.ModelProto()
    with open(onnx_model_path, mode='rb') as f:
        content = f.read()
        onnx_model.ParseFromString(content)
    quantized_data_types = {onnx.TensorProto.UINT8, onnx.TensorProto.INT8, onnx.TensorProto.UINT16, onnx.TensorProto.INT16}
    for initializer in onnx_model.graph.initializer:
        if initializer.data_type in quantized_data_types:
            return True  # Model is quantized
    return False  # No integer initializers found, model is not quantized