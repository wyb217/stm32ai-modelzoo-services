# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

""" Implementation of Conv + LSTM denoiser model"""
import torch
import torch.nn as nn
from einops import rearrange
from . import DecomposedLSTM

class ConvLSTMDenoiser(nn.Module):
    '''
    LSTM-based recurrent denoiser model.
       Consists of a 1D convolutional encoder, a block of consecutive LSTM layers,
       and a 1D convolutional decoder.
       Takes magnitude spectrogram columns as input, and outputs a mask that should be applied
       to the complex spectrogram.
       Input shape : (batch, in_channels, sequence_length)
       Output shape : (batch, out_channels, sequence_length)
    '''
    def __init__(self, in_channels=257, out_channels=257, lstm_hidden_size=256, num_lstm_layers=2):
        '''
        Parameters
        ----------
        in_channels, int : Number of input channels, should correspond to n_fft // 2 + 1
            e.g., if n_fft = 512, in_channels = 257
            Input shape of the model is : (batch, in_channels, sequence_length)
        out_channels, int : Number of output channels, should correspond to n_fft // 2 + 1
            e.g., if n_fft = 512, out_channels = 257
            Output shape of the model is : (batch, out_channels, sequence_length)
        lstm_hidden_size, int : Number of hidden units in LSTM layers
            Corresponds to the hidden_size parameter of torch.nn.LSTM
        num_lstm_layers, int : Number of consecutive LSTM layers to include in the middle LSTM block.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = lstm_hidden_size
        self.lstm_layers = num_lstm_layers
        self.conv_block_1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                                     out_channels=self.out_channels,
                                                     kernel_size=1),
                                          nn.BatchNorm1d(num_features=self.out_channels),
                                          nn.ReLU()
                                          )
        self.lstm_block = nn.LSTM(input_size=out_channels,
                                  hidden_size=self.hidden_size,
                                  num_layers=self.lstm_layers,
                                  batch_first=True)
        self.conv_block_2 = nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                                     out_channels=self.out_channels,
                                                     kernel_size=1),
                                          nn.BatchNorm1d(num_features=self.out_channels),
                                          nn.ReLU())
        self.conv_3 = nn.Conv1d(in_channels=self.out_channels,
                                out_channels=self.out_channels,
                                kernel_size=1)
        self.activation = nn.Sigmoid()


    def forward(self, x):
        # x should be of shape (batch, in_channels, seq_length) or in our case (batch, n_fft // 2 + 1, n_frames)
        conv_1_out = self.conv_block_1(x)
        lstm_in = rearrange(conv_1_out, "b c l -> b l c")
        lstm_out, _ = self.lstm_block(lstm_in)
        conv_2_in = rearrange(lstm_out, "b l c -> b c l")
        conv_2_out = self.conv_block_2(conv_2_in)
        conv_3_out = self.conv_3(conv_2_out)
        output = self.activation(conv_3_out)

        return output
    
class DecomposedConvLSTMDenoiser(nn.Module):
    '''
    LSTM-based recurrent denoiser model.
    Uses decomposed LSTM layers to fit the STEDGEAI ATONN compiler.
    Does not take sequences as input, instead only takes a single timestep at a time.
    Consists of a 1D convolutional encoder, a block of consecutive LSTM layers,
    and a 1D convolutional decoder.
    Takes one magnitude spectrogram column as input, as well as the hidden and cell states 
    of the LSTM layers at the previous timestep.
    /!\ Experimental, very likely to change with changes to STEDGEAI /!\
    '''
    # Fixed at 2 LSTM layers to avoid introducing Slice ops to the ONNX graph, which would cause 
    # problems with some older versions of STEDGEAI
    def __init__(self, in_channels=257, out_channels=257, lstm_hidden_size=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = lstm_hidden_size
        self.conv_block_1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                                     out_channels=self.out_channels,
                                                     kernel_size=1),
                                          nn.BatchNorm1d(num_features=self.out_channels),
                                          nn.ReLU()
                                          )
        self.lstm_1 = DecomposedLSTM(input_size=self.in_channels,
                                hidden_size=self.hidden_size,
                                bias=True,
                                separate_weights=True,
                                batch_first=True)
        self.lstm_2 = DecomposedLSTM(input_size=self.hidden_size,
                                hidden_size=self.hidden_size,
                                bias=True,
                                separate_weights=True,
                                batch_first=True)
        
        self.conv_block_2 = nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                                     out_channels=self.out_channels,
                                                     kernel_size=1),
                                          nn.BatchNorm1d(num_features=self.out_channels),
                                          nn.ReLU())
        self.conv_3 = nn.Conv1d(in_channels=self.out_channels,
                                out_channels=self.out_channels,
                                kernel_size=1)
        self.activation = nn.Sigmoid()


    def forward(self, x, h_prev_1, h_prev_2, c_prev_1, c_prev_2):
        # x should be of shape (batch, in_channels, seq_length) or in our case (n_fft // 2 + 1, n_frames)
        conv_1_out = self.conv_block_1(x)
        lstm_in = rearrange(conv_1_out, "b c l -> b l c")
        h_1, c_1 = self.lstm_1(lstm_in, h_prev_1, c_prev_1)
        # Reintroduce dummy timestep dimension
        lstm_2_in = torch.unsqueeze(h_1, 1)
        h_2, c_2 = self.lstm_2(lstm_2_in, h_prev_2, c_prev_2)
        conv_2_in = rearrange(h_2, "1 c -> 1 c 1")
        conv_2_out = self.conv_block_2(conv_2_in)
        conv_3_out = self.conv_3(conv_2_out)
        output = self.activation(conv_3_out)

        return output, h_1, h_2, c_1, c_2
    
class DecomposedConvLSTMDenoiserWithSlices(nn.Module):
    ''' Same thing as DecomposedConvLSTMDenoiser, but supports arbitrary number of LSTM layers
        This introduces Slice nodes in the ONNX graph, which was causing STEDGEAI problems at some point.
    '''
    def __init__(self, in_channels=257, out_channels=257, lstm_hidden_size=256, num_lstm_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = lstm_hidden_size
        self.lstm_layers = num_lstm_layers
        self.conv_block_1 = nn.Sequential(nn.Conv1d(in_channels=self.in_channels,
                                                        out_channels=self.out_channels,
                                                        kernel_size=1),
                                            nn.BatchNorm1d(num_features=self.out_channels),
                                            nn.ReLU()
                                            )
        self.lstm_module_list = []
        for i in range(num_lstm_layers):
            if i == 0:
                self.lstm_module_list.append(DecomposedLSTM(input_size=self.in_channels,
                                        hidden_size=self.hidden_size,
                                        bias=True,
                                        separate_weights=True,
                                        batch_first=True))
            else:
                self.lstm_module_list.append(DecomposedLSTM(input_size=self.hidden_size,
                                        hidden_size=self.hidden_size,
                                        bias=True,
                                        separate_weights=True,
                                        batch_first=True))
                
        self.lstm_block = nn.Sequential(*self.lstm_module_list)
        self.conv_block_2 = nn.Sequential(nn.Conv1d(in_channels=self.hidden_size,
                                                        out_channels=self.out_channels,
                                                        kernel_size=1),
                                            nn.BatchNorm1d(num_features=self.out_channels),
                                            nn.ReLU())
        self.conv_3 = nn.Conv1d(in_channels=self.out_channels,
                                out_channels=self.out_channels,
                                kernel_size=1)
        self.activation = nn.Sigmoid()


    def forward(self, x, h_prev, c_prev):
        # x should be of shape (batch, in_channels, seq_length) or in our case (n_fft // 2 + 1, n_frames)
        conv_1_out = self.conv_block_1(x)
        lstm_in = rearrange(conv_1_out, "b c l -> b l c")
        h_list = []
        c_list = []
        for i, layer in enumerate(self.lstm_block):
            if i == 0:
                h, c = layer(lstm_in, h_prev[i], c_prev[i])
            else:
                h, c = layer(h, h_prev[i], c_prev[i])
            h_list.append(h)
            c_list.append(c)
        print(f"lstm block output shape{h.shape}")
        conv_2_in = rearrange(h, "1 c -> 1 c 1")
        conv_2_out = self.conv_block_2(conv_2_in)
        conv_3_out = self.conv_3(conv_2_out)
        output = self.activation(conv_3_out)
        h_out = torch.cat(h_list, dim=0)
        c_out = torch.cat(c_list, dim=0)
        return output, h_out, c_out
