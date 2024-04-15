import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch



def map_function_ADC(bitline_current):
    
    # linear approximation of I_total.  
    ADC_energy = torch.sum(7e-05 * bitline_current) + 0.0098 * torch.sum(bitline_current)  
    return ADC_energy

def map_function(bitline_current):
    
    # linear approximation of I_total.csv
    # Note, we are looping batch dimension
    # FC: tile_bitline_current's dimension is (activation precision, bitline index, row index of crossbar, col index of crossbar)
    # General Conv: tile_bitline_current's dimension is (activation precision, conv_sequence, bitline index, row index of crossbar, col index of crossbar)
    # Depth Conv: tile_bitline_current's dimension is (activation precision, input_channel, conv_sequence, bitline index, row index of crossbar, col index of crossbar)
    if len(bitline_current.shape) == 4: # fc
       activation_precision, crossbar_size, num_row_tile, num_col_tile = bitline_current.shape
       num_bitline_access = activation_precision * crossbar_size * num_row_tile * num_col_tile
    elif len(bitline_current.shape) == 5: # general conv
       activation_precision, num_conv_sequence, crossbar_size, num_row_tile, num_col_tile = bitline_current.shape
       num_bitline_access = activation_precision * num_conv_sequence * crossbar_size * num_row_tile * num_col_tile
    elif len(bitline_current.shape) == 6: # depth conv
       activation_precision, num_input_channel, num_conv_sequence, crossbar_size, num_row_tile, num_col_tile = bitline_current.shape
       num_bitline_access = activation_precision * num_conv_sequence * crossbar_size * num_row_tile * num_col_tile

    # use either of it top: old mapping, bottom new mapp    
    # ADC_energy = np.sum(7.15e-05 * bitline_current) + 9.79e-3 * num_bitline_access  
    ADC_energy = np.sum(6.02e-05 * bitline_current) + 6.72e-4 * num_bitline_access  
    return ADC_energy
