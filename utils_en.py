import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from map import *

def torch_dec2bin(w_int, precision=8): # -128 ~ 127
    if precision == 8:
        bin_w_int8_7 = torch.div(w_int - 127, -128, rounding_mode="trunc")
        bin_w_int8_6 = torch.div(w_int - (-128 * bin_w_int8_7), 64, rounding_mode="trunc")
        bin_w_int8_5 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6), 32, rounding_mode="trunc")
        bin_w_int8_4 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5), 16, rounding_mode="trunc")
        bin_w_int8_3 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4), 8, rounding_mode="trunc")
        bin_w_int8_2 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3), 4, rounding_mode="trunc")
        bin_w_int8_1 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3 + 4 * bin_w_int8_2), 2, rounding_mode="trunc")
        bin_w_int8_0 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3 + 4 * bin_w_int8_2 + 2 * bin_w_int8_1), 1, rounding_mode="trunc")
        return bin_w_int8_7 +  bin_w_int8_6 +  bin_w_int8_5 +  bin_w_int8_4 +  bin_w_int8_3 +  bin_w_int8_2 +  bin_w_int8_1 +  bin_w_int8_0

    elif precision == 6:
        bin_w_int6_5 = torch.div(w_int - 31, -32, rounding_mode="trunc")
        bin_w_int6_4 = torch.div(w_int - (-32 * bin_w_int6_5), 16, rounding_mode="trunc")
        bin_w_int6_3 = torch.div(w_int - (-32 * bin_w_int6_5 + 16 * bin_w_int6_4), 8, rounding_mode="trunc")
        bin_w_int6_2 = torch.div(w_int - (-32 * bin_w_int6_5 + 16 * bin_w_int6_4 + 8 * bin_w_int6_3), 4, rounding_mode="trunc")
        bin_w_int6_1 = torch.div(w_int - (-32 * bin_w_int6_5 + 16 * bin_w_int6_4 + 8 * bin_w_int6_3 + 4 * bin_w_int6_2), 2, rounding_mode="trunc")
        bin_w_int6_0 = torch.div(w_int - (-32 * bin_w_int6_5 + 16 * bin_w_int6_4 + 8 * bin_w_int6_3 + 4 * bin_w_int6_2 + 2 * bin_w_int6_1), 1, rounding_mode="trunc")
        return bin_w_int6_5 +  bin_w_int6_4 +  bin_w_int6_3 +  bin_w_int6_2 +  bin_w_int6_1 +  bin_w_int6_0

    elif precision == 4:
        bin_w_int4_3 = torch.div(w_int - 7, -8, rounding_mode="trunc")
        bin_w_int4_2 = torch.div(w_int - (-8 * bin_w_int4_3), 4, rounding_mode="trunc")
        bin_w_int4_1 = torch.div(w_int - (-8 * bin_w_int4_3 + 4 * bin_w_int4_2), 2, rounding_mode="trunc")
        bin_w_int4_0 = torch.div(w_int - (-8 * bin_w_int4_3 + 4 * bin_w_int4_2 + 2 * bin_w_int4_1), 1, rounding_mode="trunc")
        return bin_w_int4_3 +  bin_w_int4_2 +  bin_w_int4_1 +  bin_w_int4_0


def torch_dec2bin_keep_dim(w_int, precision=8):
    if precision == 8:
        bin_w_int8_7 = torch.div(w_int - 127, -128, rounding_mode="trunc").unsqueeze(0)
        bin_w_int8_6 = torch.div(w_int - (-128 * bin_w_int8_7), 64, rounding_mode="trunc")
        bin_w_int8_5 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6), 32, rounding_mode="trunc")
        bin_w_int8_4 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5), 16, rounding_mode="trunc")
        bin_w_int8_3 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4), 8, rounding_mode="trunc")
        bin_w_int8_2 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3), 4, rounding_mode="trunc")
        bin_w_int8_1 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3 + 4 * bin_w_int8_2), 2, rounding_mode="trunc")
        bin_w_int8_0 = torch.div(w_int - (-128 * bin_w_int8_7 + 64 * bin_w_int8_6 + 32 * bin_w_int8_5 + 16 * bin_w_int8_4 + 8 * bin_w_int8_3 + 4 * bin_w_int8_2 + 2 * bin_w_int8_1), 1, rounding_mode="trunc")

        # (precision, output feature, input feature)
        return torch.cat((bin_w_int8_7,  bin_w_int8_6,  bin_w_int8_5,  bin_w_int8_4,  bin_w_int8_3,  bin_w_int8_2,  bin_w_int8_1,  bin_w_int8_0), dim=0)

    elif precision == 6:
        bin_w_int6_5 = torch.div(w_int - 31, -32, rounding_mode="trunc").unsqueeze(0)
        bin_w_int6_4 = torch.div(w_int - (-32 * bin_w_int6_5), 16, rounding_mode="trunc")
        bin_w_int6_3 = torch.div(w_int - (-32 * bin_w_int6_5 + 16 * bin_w_int6_4), 8, rounding_mode="trunc")
        bin_w_int6_2 = torch.div(w_int - (-32 * bin_w_int6_5 + 16 * bin_w_int6_4 + 8 * bin_w_int6_3), 4, rounding_mode="trunc")
        bin_w_int6_1 = torch.div(w_int - (-32 * bin_w_int6_5 + 16 * bin_w_int6_4 + 8 * bin_w_int6_3 + 4 * bin_w_int6_2), 2, rounding_mode="trunc")
        bin_w_int6_0 = torch.div(w_int - (-32 * bin_w_int6_5 + 16 * bin_w_int6_4 + 8 * bin_w_int6_3 + 4 * bin_w_int6_2 + 2 * bin_w_int6_1), 1, rounding_mode="trunc")
        return torch.cat((bin_w_int6_5,  bin_w_int6_4,  bin_w_int6_3,  bin_w_int6_2,  bin_w_int6_1,  bin_w_int6_0), dim=0)

    elif precision == 4:
        bin_w_int4_3 = torch.div(w_int - 7, -8, rounding_mode="trunc").unsqueeze(0)
        bin_w_int4_2 = torch.div(w_int - (-8 * bin_w_int4_3), 4, rounding_mode="trunc").unsqueeze(0)
        bin_w_int4_1 = torch.div(w_int - (-8 * bin_w_int4_3 + 4 * bin_w_int4_2), 2, rounding_mode="trunc").unsqueeze(0)
        bin_w_int4_0 = torch.div(w_int - (-8 * bin_w_int4_3 + 4 * bin_w_int4_2 + 2 * bin_w_int4_1), 1, rounding_mode="trunc").unsqueeze(0)
        return torch.cat((bin_w_int4_3,  bin_w_int4_2,  bin_w_int4_1,  bin_w_int4_0), dim=0)


# def map_to_energy():

    

def tile_bitline_current_conv(input, layer, precision=8, crossbar_size=256, depth_conv=False):
    # input: (batch, input channel, height, width), weight: (num output filters, filter depth, filter height, filter width)
    batch_size = input.shape[0]
    device = input.device
    # input = input.detach().cpu()
    input = input.detach()
    bin_input = torch_dec2bin_keep_dim(input / input.scale + input.zero_point, precision=precision) # (activation precision, batch, input channel, input height, input width)
    bin_weight = torch_dec2bin_keep_dim(layer.quant_weight().value / layer.quant_weight().scale + layer.quant_weight().zero_point, precision=precision) # (weight bit, num filters, filter depth, filter height, filter width)
    #bin_input = torch_dec2bin_keep_dim(input, precision=precision) # (activation bit, batch, input feature)
    #bin_weight = torch_dec2bin_keep_dim(layer, precision=precision) # (weight bit, output feature, input feature)
    bin_weight = bin_weight.reshape(precision * bin_weight.shape[1], bin_weight.shape[2] * bin_weight.shape[3] * bin_weight.shape[4]) # (weight bit * output feature, input feature)
    bin_weight = bin_weight.transpose(1, 0) # (input feature, weight bit * output feature)


    # Now, let's make bin_weight matrix same as weight mapping in PIM
    if (bin_weight.shape[0] <= crossbar_size) and (bin_weight.shape[1] <= crossbar_size):
        num_row_tile, num_col_tile = 1, 1
        tile_bin_weight = torch.zeros((crossbar_size, crossbar_size, num_row_tile, num_col_tile), requires_grad=False)#.to(device)
        tile_bin_weight[:bin_weight.shape[0], :bin_weight.shape[1],0,0] = bin_weight
    else:
        if bin_weight.shape[0] % crossbar_size == 0:
            num_row_tile = (bin_weight.shape[0] // crossbar_size)
        else:
            num_row_tile = (bin_weight.shape[0] // crossbar_size) + 1

        if bin_weight.shape[1] % crossbar_size == 0:
            num_col_tile = (bin_weight.shape[1] // crossbar_size)
        else:
            num_col_tile = (bin_weight.shape[1] // crossbar_size) + 1

        tile_bin_weight = torch.zeros((num_row_tile * crossbar_size, num_col_tile * crossbar_size), requires_grad=False)#.to(device)
        tile_bin_weight[:bin_weight.shape[0], :bin_weight.shape[1]] = bin_weight

        tile_bin_weight = F.unfold(tile_bin_weight.unsqueeze(0), kernel_size=crossbar_size, stride=crossbar_size)
        tile_bin_weight = tile_bin_weight.reshape(crossbar_size, crossbar_size, num_row_tile, num_col_tile)


    # Now, let's make bin_input matrix same as input sequence (convolution sequence) in PIM
    #conv_sequence_height = int(np.floor((input.shape[-2] + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) / layer.stride[0] + 1))
    #conv_sequence_width = int(np.floor((input.shape[-1] + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) / layer.stride[1] + 1))


    if True:
        if depth_conv == False:
            if layer.padding == "same":
                # Please modify padding depending on layers
                conv_bin_input = F.unfold(bin_input.reshape(precision * batch_size, bin_input.shape[2], bin_input.shape[3], bin_input.shape[4]), kernel_size=layer.kernel_size, stride=layer.stride, padding=(1,1))
            else:
                conv_bin_input = F.unfold(bin_input.reshape(precision * batch_size, bin_input.shape[2], bin_input.shape[3], bin_input.shape[4]), kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
            conv_bin_input = conv_bin_input.reshape(precision, batch_size, conv_bin_input.shape[1], conv_bin_input.shape[2]).transpose(2,3)

            tile_bin_input = torch.zeros((precision, batch_size, conv_bin_input.shape[-2], crossbar_size, num_row_tile), requires_grad=False)#.to(device)
            if num_row_tile == 1:
                tile_bin_input[:,:,:,:conv_bin_input.shape[-1],0] = conv_bin_input
            elif num_row_tile > 1:
                # print(num_row_tile,conv_bin_input.shape, bin_weight.shape)
                for i in range(num_row_tile-1):
                    tile_bin_input[:,:,:,:,i] = conv_bin_input[:,:,:,crossbar_size*i:crossbar_size*(i+1)]
                tile_bin_input[:,:,:,:conv_bin_input.shape[-1]-crossbar_size*(i+1),-1] = conv_bin_input[:,:,:,crossbar_size*(i+1):]

            tile_bitline_current = torch.zeros((precision, batch_size, conv_bin_input.shape[-2], crossbar_size, num_row_tile, num_col_tile))
            for i in range(num_row_tile):
                for j in range(num_col_tile):
                    tile_bitline_current[:,:,:,:,i,j] = torch.matmul(tile_bin_input[:,:,:,:,i].reshape(precision, batch_size * conv_bin_input.shape[-2], crossbar_size), \
                                                                tile_bin_weight[:,:,i,j]).reshape(precision, batch_size, conv_bin_input.shape[-2], crossbar_size)

            '''
            print("input, weight: ", input.shape, layer.weight.data.shape)
            print("bin_input, bin_weight: ", bin_input.shape, bin_weight.shape)
            print("conv_bin_input: ", conv_bin_input.shape)
            print("tile_bin_input, tile_bin_weight: ", tile_bin_input.shape, tile_bin_weight.shape)
            print("tile_bitline_current: ", tile_bitline_current.shape)
            '''
            val = 0
            for b in range(0, batch_size):
                val += np.sum(map_to_energy(tile_bitline_current[:,b,:,:,:,:].detach().cpu().numpy()))

        else:

            for input_channel_idx in range(bin_input.shape[2]): # input channel
                temp_bin_input = bin_input[:,:,input_channel_idx,:,:].unsqueeze(2)

                if layer.padding == "same":
                    # Please modify padding depending on layers
                    conv_bin_input = F.unfold(temp_bin_input.reshape(precision * batch_size, temp_bin_input.shape[2], temp_bin_input.shape[3], temp_bin_input.shape[4]), kernel_size=layer.kernel_size, stride=layer.stride, padding=(1,1))
                else:
                    conv_bin_input = F.unfold(temp_bin_input.reshape(precision * batch_size, temp_bin_input.shape[2], temp_bin_input.shape[3], temp_bin_input.shape[4]), kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding)
                conv_bin_input = conv_bin_input.reshape(precision, batch_size, conv_bin_input.shape[1], conv_bin_input.shape[2]).transpose(2,3)
                
                if input_channel_idx == 0:
                    tile_bitline_current = torch.zeros((precision, batch_size, bin_input.shape[2], conv_bin_input.shape[-2], crossbar_size, num_row_tile, num_col_tile))

                # make conv_bin_input to be structured by crossbar size
                tile_bin_input = torch.zeros((precision, batch_size, conv_bin_input.shape[-2], crossbar_size, num_row_tile), requires_grad=False)#.to(device)
                if num_row_tile == 1:
                    tile_bin_input[:,:,:,:conv_bin_input.shape[-1],0] = conv_bin_input
                elif num_row_tile > 1:
                    for i in range(num_row_tile-1):
                        tile_bin_input[:,:,:,:,i] = conv_bin_input[:,:,:,crossbar_size*i:crossbar_size*(i+1)]
                    tile_bin_input[:,:,:,:conv_bin_input.shape[-1]-crossbar_size*(i+1),-1] = conv_bin_input[:,:,:,crossbar_size*(i+1):]

                temp_tile_bitline_current = torch.zeros((precision, batch_size, conv_bin_input.shape[-2], crossbar_size, num_row_tile, num_col_tile))
                for i in range(num_row_tile):
                    j = (input_channel_idx * precision) // crossbar_size #col_tile_idx

                    mask = torch.zeros_like(tile_bin_weight[:,:,i,j])
                    if (input_channel_idx * precision - j * crossbar_size) < 0:
                        mask[:conv_bin_input.shape[-1],:(input_channel_idx+1) * precision - j * crossbar_size] = 1
                    elif ((i+1) * precision - j * crossbar_size) > crossbar_size:
                        mask[:conv_bin_input.shape[-1], input_channel_idx * precision - j * crossbar_size:] = 1
                    else:
                        mask[:conv_bin_input.shape[-1], input_channel_idx * precision - j * crossbar_size:(input_channel_idx+1) * precision - j * crossbar_size] = 1

                    temp_tile_bitline_current[:,:,:,:,i,j] = torch.matmul(tile_bin_input[:,:,:,:,i].reshape(precision, batch_size * conv_bin_input.shape[-2], crossbar_size), \
                                                                tile_bin_weight[:,:,i,j] * mask).reshape(precision, batch_size, conv_bin_input.shape[-2], crossbar_size)

                tile_bitline_current[:,:,input_channel_idx,:,:,:,:] = temp_tile_bitline_current
                val = 0
                for b in range(0, batch_size):
                    val += np.sum(map_to_energy(tile_bitline_current[:,b,:,:,:,:,:].detach().cpu().numpy()))


                """
                f = plt.figure(figsize=(5,5))
                plt.imshow((tile_bin_weight[:,:,i,j] * mask).detach().cpu())
                plt.show()
                plt.savefig("mask.png")
                plt.close()
                """

            '''
            print("input, weight: ", input.shape, layer.weight.data.shape)
            print("bin_input, bin_weight: ", bin_input.shape, bin_weight.shape)
            print("conv_bin_input: ", conv_bin_input.shape)
            print("tile_bin_input, tile_bin_weight: ", tile_bin_input.shape, tile_bin_weight.shape)
            print("tile_bitline_current: ", tile_bitline_current.shape)
            '''
    # using final output, we can find a bitline current of each colum in each crossbar for each binary input sequence.
    # General Conv: tile_bitline_current's dimension is (activation precision, batch_size, conv_sequence, bitline index, row index of crossbar, col index of crossbar)
    # Depth Conv: tile_bitline_current's dimension is (activation precision, batch_size, input_channel, conv_sequence, bitline index, row index of crossbar, col index of crossbar)

    # Note that, in general conv, all the input channels are processed by a 3D kernel, but
    #            in depth conv, each input channel is processed by a 2D kernel.

    # [General Conv] For example, to find a i-th bitline's current, in j-th row and k-th col crossbar, for a-th activation (binary), b-th batch input, and c-th convolution sequence
    # access -> tile_biltline_current[a, b, c, i, j, k]

    # [Depth Conv] For example, to find a i-th bitline's current for x-th input channel, in j-th row and k-th col crossbar, for a-th activation (binary), b-th batch input, and c-th convolution sequence
    # access -> tile_biltline_current[a, b, x, c, i, j, k]
    # exit()
    
    # return np.sum(map_to_energy(tile_bitline_current.flatten().numpy()))
    return val


def tile_bitline_current_fc(input, layer, precision=8, crossbar_size=256):
    # input: (batch, input feature), weight: (output feature, input feature)
    batch_size = input.shape[0]
    device = input.device

    bin_input = torch_dec2bin_keep_dim(input / input.scale + input.zero_point, precision=precision) # (activation precision, batch, input feature)
    bin_weight = torch_dec2bin_keep_dim(layer.quant_weight().value / layer.quant_weight().scale + layer.quant_weight().zero_point, precision=precision) # (weight bit, output feature, input feature)
    #bin_input = torch_dec2bin_keep_dim(input, precision=precision) # (activation bit, batch, input feature)
    #bin_weight = torch_dec2bin_keep_dim(layer, precision=precision) # (weight bit, output feature, input feature)
    bin_weight = bin_weight.reshape(precision * bin_weight.shape[1], bin_weight.shape[2]) # (weight bit * output feature, input feature)
    bin_weight = bin_weight.transpose(1, 0) # (input feature, weight bit * output feature)


    # Now, let's make bin_weight matrix same as weight mapping in PIM
    if (bin_weight.shape[0] <= crossbar_size) and (bin_weight.shape[1] <= crossbar_size):
        num_row_tile, num_col_tile = 1, 1
        tile_bin_weight = torch.zeros((crossbar_size, crossbar_size, num_row_tile, num_col_tile)).to(device)
        tile_bin_weight[:bin_weight.shape[0], :bin_weight.shape[1],0,0] = bin_weight
    else:
        if bin_weight.shape[0] % crossbar_size == 0:
            num_row_tile = (bin_weight.shape[0] // crossbar_size)
        else:
            num_row_tile = (bin_weight.shape[0] // crossbar_size) + 1

        if bin_weight.shape[1] % crossbar_size == 0:
            num_col_tile = (bin_weight.shape[1] // crossbar_size)
        else:
            num_col_tile = (bin_weight.shape[1] // crossbar_size) + 1

        tile_bin_weight = torch.zeros((num_row_tile * crossbar_size, num_col_tile * crossbar_size)).to(device)
        tile_bin_weight[:bin_weight.shape[0], :bin_weight.shape[1]] = bin_weight

        tile_bin_weight = F.unfold(tile_bin_weight.unsqueeze(0), kernel_size=crossbar_size, stride=crossbar_size)
        tile_bin_weight = tile_bin_weight.reshape(crossbar_size, crossbar_size, num_row_tile, num_col_tile)


    # Now, let's make bin_input matrix same as input sequence in PIM
    if (bin_input.shape[-1] <= crossbar_size):
        num_row_tile = 1
        tile_bin_input = torch.zeros((precision, batch_size, crossbar_size, num_row_tile)).to(device)
        tile_bin_input[:,:,:bin_input.shape[-1],0] = bin_input
    else:
        if bin_input.shape[-1] % crossbar_size == 0:
            num_row_tile = (bin_input.shape[-1] // crossbar_size)
        else:
            num_row_tile = (bin_input.shape[-1] // crossbar_size) + 1

        tile_bin_input = torch.zeros((precision, batch_size, num_row_tile * crossbar_size)).to(device)
        tile_bin_input[:, :, :bin_input.shape[-1]] = bin_input

        tile_bin_input = F.unfold(tile_bin_input, kernel_size=(batch_size, crossbar_size), stride=(batch_size, crossbar_size))
        tile_bin_input = tile_bin_input.reshape(precision, batch_size, crossbar_size, num_row_tile)


    # calculate the crossbar-wise bitline current
    tile_bitline_current = torch.zeros((precision, batch_size, crossbar_size, num_row_tile, num_col_tile))
    for i in range(num_row_tile):
        for j in range(num_col_tile):
            tile_bitline_current[:,:,:,i,j] = torch.matmul(tile_bin_input[:,:,:,i], tile_bin_weight[:,:,i,j])

    '''
    print("input, weight: ", input.shape, layer.weight.data.shape)
    print("bin_input, bin_weight: ", bin_input.shape, bin_weight.shape)
    print("tile_bin_input, tile_bin_weight: ", tile_bin_input.shape, tile_bin_weight.shape)
    print("tile_bitline_current: ", tile_bitline_current.shape)
    '''
    #exit()

    # using final output, we can find a bitline current of each colum in each crossbar for each binary input sequence.
    # tile_bitline_current's dimension is (activation precision, batch_size, bitline index, row index of crossbar, col index of crossbar)

    # For example, to find a i-th bitline's current in j-th row and k-th col crossbar for a-th activation and b-th batch input
    # access -> tile_biltline_current[a, b, i, j, k]
    # return torch.sum(tile_bitline_current)
    # return np.sum(map_to_energy(tile_bitline_current.flatten().numpy()))
    val = 0
    for b in range(0, batch_size):
        val += np.sum(map_to_energy(tile_bitline_current[:,b,:,:,:].detach().cpu().numpy()))
        # val += np.sum((tile_bitline_current[:,b,:,:,:].flatten().detach().cpu().numpy()))

    return val


if __name__ == '__main__':
    print("here")
    x = torch.tensor([[[1., 0., 1.],
                       [0., 0., 1.],
                       [1., 1., 0.]],
                      [[0., 1., 0.],
                       [1., 1., 1.],
                       [0., 1., 0.]],
                      [[0., 1., 1.],
                       [1., 0., 0.],
                       [0., 1., 1.]]])
 
    #x = x.unsqueeze(0).repeat(3,1,1)
    layer = torch.nn.Conv2d(in_channels=3, out_channels=3, groups=3, kernel_size=2, stride=1, bias=False)
    layer.weight.data[layer.weight.data > 0] = 1
    layer.weight.data[layer.weight.data <= 0] = -1

    print("input, weight: ", x.shape, layer.weight.data.shape)
    print("weight: ", layer.weight.data)
    print("output: ", layer(x).shape)
    print("output: ", layer(x))