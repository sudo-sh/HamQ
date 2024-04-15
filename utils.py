import json
import logging
import os
import shutil
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import brevitas.nn as qnn
# import brevitas.function as BF
import brevitas.quant as BQ
import matplotlib.pyplot as plt
import seaborn as sns

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
    
  
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)




def accuracy(outputs, labels):    
    _, predicted = torch.max(outputs.tensor, 1) # works for quantized model
    #_, predicted = torch.max(outputs, 1) # wors for FP model
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total



def accuracy_normal(outputs, labels):
        
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()
    return 100 * correct / total



def get_quant_out(x):
    quant_output = x
    output_tensor = quant_output.tensor
    scale = quant_output.scale
    zero_point = quant_output.zero_point
    int_output = torch.round(output_tensor / scale + zero_point)
    return int_output



def plot_heatmap(binary_matrix, save_path):
    matrix = np.array([[int(bit) for bit in string] for row in binary_matrix for string in row])
    matrix = matrix.reshape(binary_matrix.shape[0], -1)

    # Set a custom color palette
    colors = ["#ffffff", "#1f77b4"]  # White for 0, Blue for 1
    cmap = sns.color_palette(colors)

    # Create a grid of subplots with 2 rows and 1 column
    fig, axs = plt.subplots(2, 1, figsize=(20, 40), gridspec_kw={'height_ratios': [30, 5]})

    # Plot the matrix heatmap in the top subplot
    ax_heatmap = axs[0]
    sns.heatmap(matrix, cmap=cmap, cbar=False, linewidths=0.3, linecolor='gray', square=True, annot=True, fmt="d", ax=ax_heatmap)
    ax_heatmap.set_xlabel("X-axis")
    ax_heatmap.set_ylabel("Y-axis")
    ax_heatmap.set_title("Matrix Heatmap")

    # Compute the number of 1's in each column of the matrix
    ones_count = np.sum(matrix, axis=0)

    # Plot the number of 1's in the bottom subplot
    ax_ones_count = axs[1]
    ax_ones_count.bar(range(len(ones_count)), ones_count)
    ax_ones_count.set_xlabel("Column Index")
    ax_ones_count.set_ylabel("Number of 1's")
    ax_ones_count.set_title("Number of 1's in Each Column")

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)

def plot_layer_matrix_heatmap(model, layer_name, bit_quant, stamp):
    weights_signed = getattr(model, layer_name).int_weight().to(torch.int8)
    binary_str_tensor = get_binary_representation(weights_signed, bit_quant)
    binary_str_tensor = binary_str_tensor.reshape(weights_signed.size(-1), -1)
    # print(binary_str_tensor[0])
    plot_heatmap(binary_str_tensor, "fig/"+layer_name+"_heatmap_"+stamp+".jpg")




def binary_torch(x, bits):
    mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def get_power(model, bit_quant, layer_names):
    # layer_names = ['fc1', 'fc2', 'conv1', 'conv2']
    total_ones = 0

    for layer_name in layer_names:
        total_ones_layer = 0
        params = 0
        for name, param in model.named_parameters():
            if layer_name in name and 'weight' in name:
                weights_signed = getattr(model, layer_name).int_weight().to(torch.int8)
                # total_ones_layer = get_ones(weights_signed, bit_quant)
                bin_hw = binary_torch(weights_signed, bit_quant)
                # print(bin_hw.size())
                hw_ = bin_hw.sum()
                total_ones_layer = hw_
                params = weights_signed.view(-1).size(0)
                break

        logging.info("{} params: {}; {} ones: {} ; percentage: {}".format(layer_name, params, layer_name, total_ones_layer, total_ones_layer/(params*bit_quant)))
        total_ones += total_ones_layer

    return total_ones

def get_num_zero_weights(model):
    num_zero_weights = 0
    total_params = 0

    # Iterate over the model's parameters
    for name, param in model.named_parameters():
        if 'weight' in name:  # Consider only weight parameters
            num_zero_weights += ((param < 1e-8) & (param > -1e-8)).sum().item()
            total_params += param.numel()

    print("num_zero_params", num_zero_weights)
    print("total_params",total_params)
    # exit()
    # Calculate the proportion of zero weights
    zero_weights_ratio = num_zero_weights / total_params

    return zero_weights_ratio


# Function to calculate Hamming weight of a binary string
def hamming_weight_ste(binary_string):
    return sum(bit == '1' for bit in binary_string)


class StraightThroughHW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, layer_name, model, device, params):
        '''
        # Compute the Hamming weight of the binary input
        # output = torch.sum(input != 0, dim=1).float()
        weights_signed = getattr(model, layer_name).int_weight().to(torch.int8)
        binary_str_tensor = get_binary_representation(weights_signed, params.bit_quant)
        binary_strings = binary_str_tensor.reshape(input.size(0), -1)
        # Calculate Hamming weight for each element in the array
        hamming_weights = np.vectorize(hamming_weight_ste)(binary_strings)
        # print(hamming_weights.shape)
        
        hw_ = torch.from_numpy(hamming_weights).to(torch.float64).to(device)
        
        '''
        weights_signed = getattr(model, layer_name).int_weight().to(torch.int8)
        weights_signed = weights_signed.view(weights_signed.size(0), -1)
        bin_hw = binary_torch(weights_signed, params.bit_quant)
        # print(bin_hw.size())
        hw_ = bin_hw.sum(dim = 2)
        
        threshold = params.hw_range

        # Create mask array based on the threshold
        mask = torch.where(hw_ < threshold, torch.tensor(1, device=hw_.device), torch.tensor(-1, device=hw_.device))
        
        grad_hw = mask * hw_
        # print(grad_hw)
        # exit()
        ctx.save_for_backward(input, grad_hw)
        
        return torch.abs(input) + hw_
        # return  input + hw_ - input.detach()
        
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve the saved binary input from the context
        input, hw_ = ctx.saved_tensors
        # print("Gradient")
        # print(grad_output[0])
        grad_input = grad_output.clone()
        grad_input = grad_input*hw_
        # print(grad_input[0])
        # exit()
        # Return the gradient with respect to the input
        return grad_input, None, None, None, None
    

def local_weighted_loss(model, params, device):
    penalty_factor = params.weight_penalty
    loss = 0.0

    for name, param in model.named_parameters():
        if "weight" in name:
            if "layer" in name:
                # convolution in BasicBlock or Bottleneck in ResNet
                if "conv" in name:
                    param = param.view(param.size(0),-1)
                    distances = StraightThroughHW.apply(param, name.split(".")[1], getattr(model, name.split(".")[0]), device, params) - torch.abs(param)
                    loss += penalty_factor * distances.sum()
                elif "downsample.0" in name:
                    param = param.view(param.size(0),-1)
                    distances = StraightThroughHW.apply(param, name.split(".")[2], getattr(getattr(model, name.split(".")[0]), name.split(".")[1]), device, params) - torch.abs(param)
                    loss += penalty_factor * distances.sum()
            else:
                # convolution or fc outside BasicBlock and Bottleneck in ResNet
                if ("conv" in name):
                    param = param.view(param.size(0),-1)
                    distances = StraightThroughHW.apply(param, name.split(".")[0], model, device, params) - torch.abs(param)
                    loss += penalty_factor * distances.sum()
                elif "fc" in name:
                    distances = StraightThroughHW.apply(param, name.split(".")[0], model, device, params) - torch.abs(param)
                    loss += penalty_factor * distances.sum()

        """ Sudarshan's implementation
        for layer_name in layer_names:
            if layer_name in name and 'weight' in name:
                if("conv" in layer_name):
                    param = param.view(param.size(0),-1)
                # print("layer", layer_name)
                # print("model_name", name)
                # print("param", param.size())
                # print(param.size())
                # exit()
                # quant_weight = getattr(model, layer_name).quant_weight()
                # distances = reg_local_loss_HW(quant_weight.scale, quant_weight.zero_point, param, params, device)
                distances = StraightThroughHW.apply(param, layer_name, model, device, params) - torch.abs(param)
                loss += penalty_factor * distances.sum()
        """
    # exit()
    return loss


def torch_dec2bin(w_int8):
    bin_w_int8_7 = torch.div(w_int8 - 127, -128, rounding_mode="trunc")
    bin_w_int8_6 = torch.div(w_int8 - (-128 * bin_w_int8_7), 64, rounding_mode="trunc")
    bin_w_int8_5 = torch.div(w_int8 - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6), 32, rounding_mode="trunc")
    bin_w_int8_4 = torch.div(w_int8 - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5), 16, rounding_mode="trunc")
    bin_w_int8_3 = torch.div(w_int8 - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4), 8, rounding_mode="trunc")
    bin_w_int8_2 = torch.div(w_int8 - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3), 4, rounding_mode="trunc")
    bin_w_int8_1 = torch.div(w_int8 - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3 + 4 * bin_w_int8_2), 2, rounding_mode="trunc")
    bin_w_int8_0 = torch.div(w_int8 - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3 + 4 * bin_w_int8_2 + 2 * bin_w_int8_1), 1, rounding_mode="trunc")
    
    return bin_w_int8_7 + bin_w_int8_6 + bin_w_int8_5 + bin_w_int8_4 + bin_w_int8_3 + bin_w_int8_2 + bin_w_int8_1 + bin_w_int8_0




def lx_reg_loss(model, params, layer_names, norm = 1):
    penalty_factor = params.weight_penalty
    loss = 0.0

    for name, param in model.named_parameters():
        for layer_name in layer_names:
            if layer_name in name and 'weight' in name:
                quant_weight = getattr(model, layer_name).quant_weight()
                distances = torch.norm(quant_weight - getattr(model, layer_name).quant_weight().zero_point, p =norm)
                loss += penalty_factor * distances.sum()

    return loss

