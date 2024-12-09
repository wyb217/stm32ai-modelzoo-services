# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from pathlib import Path
import copy
import onnx
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat, QuantType
from quantization import DataLoaderDataReader
import preprocessing
from dataset_utils import load_dataset_from_cfg
from torch.utils.data import DataLoader
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed

def _quantize(float_model_path,
              output_dir,
              quant_dl,
              extra_opsets=None,
              op_types_to_quantize=None,
              calibrate_method="MinMax",
              per_channel=True,
              reduce_range=False,
              extra_options=None,
              static_axis_name="seq_len",
              static_sequence_length=26
              ):
    '''Quantizes a float ONNX model. Only accepts '''

    print("[INFO] Loading ONNX model for quantization")
    float_onnx_model = onnx.load(float_model_path)

    # Get input/output node names
    # Output node name is unused for now
    output_nodes =[node.name for node in float_onnx_model.graph.output]
    input_all = [node.name for node in float_onnx_model.graph.input]
    input_initializer =  [node.name for node in float_onnx_model.graph.initializer]
    net_input_nodes = list(set(input_all)  - set(input_initializer))

    # print(f"Input node(s) : {net_input_nodes}")
    # print(f"Output node(s) : {output_nodes}")

    # If model has multiple inputs, throw an unsupported error
    # Will add support for multi-input models w/ decomposed LSTM support
    if len(net_input_nodes) > 1:
        raise NotImplementedError("Multi-input models are currently unsupported for quantization in the zoo")

    data_reader = DataLoaderDataReader(quant_dl=quant_dl,
                                       input_name=net_input_nodes[0],
                                       replace_dl_collate=False)
    
    # Preprocess the float model
    
    onnx_prep_path = Path(output_dir, "preprocessed_model.onnx") 
    
    quant_pre_process(input_model=float_model_path, output_model_path=onnx_prep_path)
    print(f"[INFO] Saved preprocessed float ONNX model at {onnx_prep_path}")

    onnx_prep_model = onnx.load(onnx_prep_path)

    # Get model's opset version
    main_opset_version = onnx_prep_model.opset_import[0].version
    # Remove superfluous ONNX opsets added by the preprocessing step
    del onnx_prep_model.opset_import[:]
    opset = onnx_prep_model.opset_import.add()
    opset.domain = '' 
    opset.version = main_opset_version

    # Add extra opsets specified by user
    if extra_opsets:
        for dom, value in extra_opsets.items():
            opset = onnx_prep_model.opset_import.add()
            opset.domain = dom
            opset.version = value

    print("Opset imports after cleanup :")
    for opset in onnx_prep_model.opset_import:
        print("opset domain=%r version=%r" % (opset.domain, opset.version))

    onnx.save(onnx_prep_model, onnx_prep_path)

    # Call quantize_static

    quantized_model_path = Path(output_dir, "quantized_model_int8.onnx")

    quantize_static(onnx_prep_path,
                    quantized_model_path,
                    data_reader,
                    op_types_to_quantize=op_types_to_quantize, # e.g. ["Conv", "LSTM"]
                    calibrate_method=calibrate_method, 
                    quant_format=QuantFormat.QDQ,
                    per_channel=per_channel,
                    weight_type=QuantType.QInt8,
                    activation_type = QuantType.QInt8,
                    reduce_range=reduce_range,
                    extra_options=extra_options) # Add extra options here
    
    quantized_static_model_path = Path(output_dir, "quantized_model_int8_static.onnx")
    quant_model = onnx.load(quantized_model_path)

    make_dim_param_fixed(quant_model.graph, param_name=static_axis_name, value=static_sequence_length)
    fix_output_shapes(quant_model),
    onnx.save(quant_model, quantized_static_model_path)

    print("[INFO] Successfully converted quantized model to static input shape")

    return quantized_model_path, quantized_static_model_path



def quantize(cfg):
    
    pipeline_args = copy.copy(cfg.preprocessing)
    del pipeline_args["pipeline_type"]

    input_pipeline = getattr(preprocessing, cfg.preprocessing.pipeline_type)(
        magnitude=True, **pipeline_args)

    # Load quantisation dataset
    val_split = cfg.dataset.num_validation_samples if cfg.dataset.num_validation_samples else 0
    quant_ds = load_dataset_from_cfg(cfg,
                                     set="train",
                                     n_clips=cfg.quantization.num_quantization_samples,
                                     val_split=0,
                                     input_pipeline=input_pipeline,
                                     target_pipeline=None,
                                     quantization=True)
    
    quant_dl = DataLoader(quant_ds,
                         batch_size=1)

    # Load model
    # If onnx model is provided in config, use it 
    # Else, look for model type and model state dict.
    # If none are provided, raise error.
    float_model_path = cfg.model.onnx_path
    
    # Load op types to quantize, calibration method and extra options
    op_types_to_quantize = cfg.quantization.op_types_to_quantize
    assert (isinstance(op_types_to_quantize, list) or op_types_to_quantize is None), "op_types_to_quantize must be a list of str or None"
    
    calibrate_method = getattr(CalibrationMethod, cfg.quantization.calibration_method)
    
    extra_options = cfg.quantization.extra_options
    # assert (extra_options is dict or extra_options is None), "extra_options must be a dict or None"
    output_dir = Path(cfg.output_dir, cfg.general.saved_models_dir)
    output_dir.mkdir(exist_ok=True)
    quantized_model_path = _quantize(float_model_path=float_model_path,
                                     output_dir=output_dir,
                                     quant_dl=quant_dl,
                                     op_types_to_quantize=op_types_to_quantize,
                                     calibrate_method=calibrate_method,
                                     per_channel=cfg.quantization.per_channel,
                                     reduce_range=cfg.quantization.reduce_range,
                                     extra_options=extra_options,
                                     static_axis_name=cfg.quantization.static_axis_name,
                                     static_sequence_length=cfg.quantization.static_sequence_length)
    
    

    return quantized_model_path
