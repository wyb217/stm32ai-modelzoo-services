# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

'''Decomposed/lowered LSTM cell'''
import torch
import torch.nn as nn
import torch.nn.init as init
from einops import rearrange

class DecomposedLSTM(nn.Module):
    '''Decomposed LSTM cell, compatible with STEGDEAI. 
    Only processes one timestep at a time.
    You can keep weights and biases separated between each gate, resulting in 8 weight/bias pairs, 
    or you can group them up, resulting in a single weight/bias pair.
    Has the same input/output shapes as torch.nn.LSTM
    /!\ Should ONLY be used for inference. You can't backprop through it anyways.'''
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 bias: bool = True,
                 separate_weights: bool = False,
                 batch_first=True
                 ):
        '''
        Parameters
        ----------
        input_size, int : Number of features in input tensor
        hidden_size, int : Number of hidden units in LSTM layer
        bias, bool : If True, use bias, if False, don't.
        separate_weights : if True, each gate is assigned 2 pairs of weight/bias tensors,
            for a total of 8 pairs.
            If False, weights for all gates are grouped into a single tensor.
            Does not have an impact on float models, but makes a big difference in quantization.
        batch_first : if True, treats first input axis as the batch axis. 
                      if False, treats second input axis as the batch axis
            Note that unlike in torch.nn.LSTM, this is True by default.
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.separate_weights = separate_weights
        self.batch_first = batch_first
        if self.separate_weights:
            # Two pairs of weights and bias for each gate
            self.gate_ii = nn.Linear(in_features=self.input_size,
                                     out_features=self.hidden_size,
                                     bias=self.bias)
            self.gate_if = nn.Linear(in_features=self.input_size,
                                     out_features=self.hidden_size,
                                     bias=self.bias)
            self.gate_ig = nn.Linear(in_features=self.input_size,
                                     out_features=self.hidden_size,
                                     bias=self.bias)
            self.gate_io = nn.Linear(in_features=self.input_size,
                                     out_features=self.hidden_size,
                                     bias=self.bias)
            
            self.gate_hi = nn.Linear(in_features=self.hidden_size,
                                     out_features=self.hidden_size,
                                     bias=self.bias)
            self.gate_hf = nn.Linear(in_features=self.hidden_size,
                                     out_features=self.hidden_size,
                                     bias=self.bias)
            self.gate_hg = nn.Linear(in_features=self.hidden_size,
                                     out_features=self.hidden_size,
                                     bias=self.bias)
            self.gate_ho = nn.Linear(in_features=self.hidden_size,
                                     out_features=self.hidden_size,
                                     bias=self.bias)
        else:
            self.gates = nn.Linear(in_features = self.input_size + self.hidden_size,
                                   out_features = self.hidden_size * 4,
                                   bias=self.bias)
    
    def forward(self, input, h_prev, c_prev):
        with torch.no_grad():
            if self.separate_weights:
                if not self.batch_first:
                    input = torch.transpose(input, 0, 1)
                input = torch.squeeze(input, dim=-2)
                if h_prev.ndim < 2:
                    h_prev = torch.unsqueeze(h_prev, dim=0)
                
                # Intermediate activations for each pair of weight/bias, 2 per gate
                act_ii = self.gate_ii(input)
                act_if = self.gate_if(input)
                act_ig = self.gate_ig(input)
                act_io = self.gate_io(input)

                act_hi = self.gate_hi(h_prev)
                act_hf = self.gate_hf(h_prev)
                act_hg = self.gate_hg(h_prev)
                act_ho = self.gate_ho(h_prev)
                
                # Activations for each gate

                i_t = nn.Sigmoid()(act_ii + act_hi)
                f_t = nn.Sigmoid()(act_if + act_hf)
                g_t = nn.Tanh()(act_ig + act_hg)
                o_t = nn.Sigmoid()(act_io + act_ho)

            else:
                # If input is not of shape (batch, seq_len, input_size), we assume it is
                #  (seq_len, batch, input_size) instead as is default for torch LSTM
                if not self.batch_first:
                    input = torch.transpose(input, 0, 1)

                input = torch.squeeze(input, dim=-2)
                if h_prev.ndim < 2:
                    h_prev = torch.unsqueeze(h_prev, dim=0)
                concatenated_inputs = torch.cat((input, h_prev), axis=-1)
                transformed_input = self.gates(concatenated_inputs)
                w_i, w_f, w_g, w_o = torch.split(transformed_input, self.hidden_size, dim=-1) 
                i_t = nn.Sigmoid()(w_i)
                f_t = nn.Sigmoid()(w_f)
                g_t = nn.Tanh()(w_g)
                o_t = nn.Sigmoid()(w_o)

            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * nn.Tanh()(c_t)

            return h_t, c_t
    
    def load_lstm_weights(self,
                          W_ih : nn.Parameter,
                          W_hh : nn.Parameter,
                          bias_ih : nn.Parameter,
                          bias_hh : nn.Parameter):
        '''Loads weights extracted from a torch.nn.LSTM layer into this layer.
           You should only use this methods on weight and bias tensors from a torch.nn.LSTM layer.
           
           Inputs
           ------
           W_ih : input/hidden weight tensor.
           W_hh : hidden/hidden weight tensor
           bias_ih : input/hidden bias tensor
           bias_hh : hidden/hidden bias tensor

           Outputs
           -------
           None
        '''
        
        if self.separate_weights:
            # Torch LSTM weights are grouped in two tensors instead of 8
            # Same with the biases.
            W_ii, W_if, W_ig, W_io = torch.split(W_ih, self.hidden_size, dim=0)
            W_hi, W_hf, W_hg, W_ho = torch.split(W_hh, self.hidden_size, dim=0)
            if self.bias:
                b_ii, b_if, b_ig, b_io = torch.split(bias_ih, self.hidden_size, dim=0)
                b_hi, b_hf, b_hg, b_ho = torch.split(bias_hh, self.hidden_size, dim=0)

            # And assign everything, convert to parameter just in case it is needed
            self.gate_ii.weight = nn.Parameter(W_ii)
            self.gate_if.weight = nn.Parameter(W_if)
            self.gate_ig.weight = nn.Parameter(W_ig)
            self.gate_io.weight = nn.Parameter(W_io)
            self.gate_hi.weight = nn.Parameter(W_hi)
            self.gate_hf.weight = nn.Parameter(W_hf)
            self.gate_hg.weight = nn.Parameter(W_hg)
            self.gate_ho.weight = nn.Parameter(W_ho)
            
            if self.bias:
                self.gate_ii.bias = nn.Parameter(b_ii)
                self.gate_if.bias = nn.Parameter(b_if)
                self.gate_ig.bias = nn.Parameter(b_ig)
                self.gate_io.bias = nn.Parameter(b_io)
                self.gate_hi.bias = nn.Parameter(b_hi)
                self.gate_hf.bias = nn.Parameter(b_hf)
                self.gate_hg.bias = nn.Parameter(b_hg)
                self.gate_ho.bias = nn.Parameter(b_ho)

        else:

            self.gates.weight = nn.Parameter(torch.cat((W_ih, W_hh), axis=-1))
            if self.bias:
                self.gates.bias = nn.Parameter(bias_ih + bias_hh)

################################################
# Helper functions for in-model LSTM replacement
################################################
def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    try:
        attrs, name = name.rsplit(".", 1)
        model = get_layer(model, attrs)
    except ValueError:
        pass
    setattr(model, name, layer)


##############################################
# In-model LSTM replacement function
##############################################

def _check_lstm_args(dropout, bidirectional, proj_size):
    if dropout != 0.0:
        raise NotImplementedError("dropout > 0 not supported for LSTM decomposition")
    if bidirectional:
        raise NotImplementedError("Bidirectional LSTM decomposition not supported")
    if proj_size != 0:
        raise NotImplementedError("LSTM with projections not supported")

def replace_lstm(model, separate_weights):
    '''
    Replaces LSTM layers inside a torch model with DecomposedLSTM layers.
    Copies weights from the torch.nn.LSTM layers to the new DecomposedLSTM layers.
    Replacement is done inplace.

    /!\ DOES NOT UPDATE THE .forward() METHOD OF THE INPUT MODULE. YOU WILL NEED TO REWRITE 
    THE .forward() METHOD TO ACCOMODATE THE DecomposedLSTM LAYERS./!\ 
    
    Inputs
    ------
    model, torch.nn.Module : Torch model with LSTM layers to replace
    separate_weights : if True, each gate is assigned 2 pairs of weight/bias tensors,
            for a total of 8 pairs.
            If False, weights for all gates are grouped into a single tensor.
            Does not have an impact on float models, but makes a big difference in quantization.

    Outputs
    -------
    None, the layers are replaced inplace
    '''
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LSTM):
            # Get current LSTM layer
            lstm = get_layer(model, name)
            # Check LSTM args
            _check_lstm_args(dropout=lstm.dropout,
                             bidirectional=lstm.bidirectional,
                             proj_size=lstm.proj_size)
            # Create new replacement layer(s) (multiple if num_layers > 1)
            replacements = []
            for k in range(lstm.num_layers):
                input_size = lstm.input_size if k == 0 else lstm.hidden_size
                rep = DecomposedLSTM(input_size=input_size,
                                     hidden_size=lstm.hidden_size,
                                     bias=lstm.bias,
                                     separate_weights=separate_weights,
                                     batch_first=lstm.batch_first)
                # Get weights from LSTM layer
                W_ih = getattr(lstm, f"weight_ih_l{k}")
                W_hh = getattr(lstm, f"weight_hh_l{k}")
                bias_ih = getattr(lstm, f"bias_ih_l{k}")
                bias_hh = getattr(lstm, f"bias_hh_l{k}")

                # Load weights in DecomposedLSTM layer
                rep.load_lstm_weights(W_ih, W_hh, bias_ih, bias_hh)
                replacements.append(rep)
            # TODO : Change this to not use nn.Sequential
            replacement = nn.Sequential(*replacements)

            # Assign replacement
            print("Replacing {} with {}".format(lstm, replacement))
            set_layer(model, name, replacement)




