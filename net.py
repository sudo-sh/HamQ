import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import brevitas.nn as qnn
# import brevitas.function as BF
import brevitas.quant as BQ

from utils import *
from utils_en import *


# for resnet libraries
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor
from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface



    

__all__ = [
    "ResNet",
    "ResNet18_Weights",
    "resnet18",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1, weight_bit_width: int = 8, act_bit_width: int = 8) -> qnn.QuantConv2d:
    """3x3 convolution with padding"""
    return qnn.QuantConv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        bias=False,
        padding=dilation,
        groups=groups,
        dilation=dilation,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        return_quant_tensor=True
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, weight_bit_width: int = 8, act_bit_width: int = 8) -> qnn.QuantConv2d:
    """1x1 convolution"""
    return qnn.QuantConv2d(
        in_planes, 
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=False,
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        return_quant_tensor=True
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        new_energy: int = 0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.bn1 = norm_layer(planes)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.conv2 = conv3x3(planes, planes, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.bn2 = norm_layer(planes)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.downsample = downsample
        self.stride = stride
        self.new_energy = new_energy
        self.weight_bit_width = weight_bit_width


    def get_energy_conv(self, x, layer_name, stride, padding):
        
        HM_energy = torch.norm(F.conv2d(input=torch_dec2bin(x / x.scale + x.zero_point),\
                                         weight=torch_dec2bin(layer_name.quant_weight().value / layer_name.quant_weight().scale +\
                                                                layer_name.quant_weight().zero_point), stride=stride, padding =padding), p=1)
        return HM_energy
    
    def get_energy_fc(self, x, layer_name):
        HM_energy = torch.norm(torch.matmul(torch_dec2bin(x / x.scale + x.zero_point), \
                                             torch_dec2bin(layer_name.quant_weight().value / layer_name.quant_weight().scale + \
                                                           layer_name.quant_weight().zero_point).transpose(1,0)), p=1)

        return HM_energy

    def forward(self, x: Tensor) -> Tensor:
        HM_weight = 0
        HM_activation = 0
        HM_energy = 0
        new_HM_energy = 0
        e = 0
        identity = x
        # print(self.new_energy)

        HM_activation += torch.sum(torch_dec2bin(x / x.scale + x.zero_point))
        HM_energy += self.get_energy_conv(x, self.conv1, stride=self.stride, padding=1)
        if(self.new_energy):
            e = tile_bitline_current_conv(x, self.conv1, precision=self.weight_bit_width, crossbar_size=256)
            new_HM_energy += e
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        HM_activation += torch.sum(torch_dec2bin(out / out.scale + out.zero_point))
        HM_energy += self.get_energy_conv(out, self.conv2, stride=1, padding=1)
        if(self.new_energy):
            e = tile_bitline_current_conv(out, self.conv2, precision=self.weight_bit_width, crossbar_size=256)
            new_HM_energy += e
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            # downsample.0 : conv1x1, downsample.1: norm_layer
            HM_activation += torch.sum(torch_dec2bin(x / x.scale + x.zero_point))
            HM_energy += self.get_energy_conv(x, self.downsample[0], stride=self.downsample[0].stride, padding=self.downsample[0].padding)
            if(self.new_energy):
                e = tile_bitline_current_conv(x, self.downsample[0], precision=self.weight_bit_width, crossbar_size=256)
                new_HM_energy += e
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        if(self.new_energy):
            HM_energy = new_HM_energy

        # print("Basic_block", HM_energy)
        return out, HM_activation, HM_energy


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        new_energy: int = 0,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.bn1 = norm_layer(width)
        self.relu1 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.conv2 = conv3x3(width, width, stride, groups, dilation, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.bn2 = norm_layer(width)
        self.relu2 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.conv3 = conv1x1(width, planes * self.expansion, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu3 = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.new_energy = new_energy
        self.weight_bit_width = weight_bit_width

    def get_energy_conv(self, x, layer_name, stride, padding):
        
        HM_energy = torch.norm(F.conv2d(input=torch_dec2bin(x / x.scale + x.zero_point),\
                                         weight=torch_dec2bin(layer_name.quant_weight().value / layer_name.quant_weight().scale +\
                                                                layer_name.quant_weight().zero_point), stride=stride, padding =padding), p=1)
        return HM_energy
    
    def get_energy_fc(self, x, layer_name):
        HM_energy = torch.norm(torch.matmul(torch_dec2bin(x / x.scale + x.zero_point), \
                                             torch_dec2bin(layer_name.quant_weight().value / layer_name.quant_weight().scale + \
                                                           layer_name.quant_weight().zero_point).transpose(1,0)), p=1)

        return HM_energy

    def forward(self, x: Tensor) -> Tensor:
        HM_weight = 0
        HM_activation = 0
        HM_energy = 0
        new_HM_energy = 0
        e = 0
        identity = x
        print("Forward Bottleneck")
        print(self.new_energy)

        HM_activation += torch.sum(torch_dec2bin(x / x.scale + x.zero_point))
        HM_energy += self.get_energy_conv(x, self.conv1, stride=1, padding=0)
        if(self.new_energy):
            e = tile_bitline_current_conv(x, self.conv1, precision=self.weight_bit_width, crossbar_size=256)
            new_HM_energy += e
            print("Energy",e)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        HM_activation += torch.sum(torch_dec2bin(out / out.scale + out.zero_point))
        HM_energy += self.get_energy_conv(out, self.conv2, stride=self.stride, padding=self.dilation)
        if(self.new_energy):
            e = tile_bitline_current_conv(out, self.conv2, precision=self.weight_bit_width, crossbar_size=256)
            new_HM_energy += e
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        HM_activation += torch.sum(torch_dec2bin(out / out.scale + out.zero_point))
        HM_energy += self.get_energy_conv(out, self.conv3, stride=1, padding=0)
        if(self.new_energy):
            e = tile_bitline_current_conv(out, self.conv3, precision=self.weight_bit_width, crossbar_size=256)
            new_HM_energy += e
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            # downsample.0 : conv1x1, downsample.1: norm_layer
            HM_activation += torch.sum(torch_dec2bin(x / x.scale + x.zero_point))
            HM_energy += self.get_energy_conv(x, self.downsample[0], stride=self.downsample[0].stride, padding=self.downsample[0].padding)
            if(self.new_energy):
                e = tile_bitline_current_conv(x, self.downsample[0], precision=self.weight_bit_width, crossbar_size=256)
                new_HM_energy += e
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        if(self.new_energy):
            HM_energy = new_HM_energy

        return out, HW_activation, HW_energy


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        new_energy: int = 0,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.new_energy = new_energy
        self.weight_bit_width = weight_bit_width


        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.quant_input = qnn.QuantIdentity(bit_width=act_bit_width, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(
            3,
            self.inplanes,
            kernel_size=7,
            stride=2, 
            padding=3,
            bias=False,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            return_quant_tensor=True,
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True, inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_relu = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True, inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0], weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, new_energy = new_energy)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, new_energy = new_energy)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, new_energy = new_energy)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], weight_bit_width=weight_bit_width, act_bit_width=act_bit_width, new_energy = new_energy)

        self.layer1_0, self.layer1_1 = self.layer1[0], self.layer1[1]
        self.layer2_0, self.layer2_1 = self.layer2[0], self.layer2[1]
        self.layer3_0, self.layer3_1 = self.layer3[0], self.layer3[1]
        self.layer4_0, self.layer4_1 = self.layer4[0], self.layer4[1]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_relu = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True, inplace=True)

        self.fc = qnn.QuantLinear(
            512 * block.expansion, 
            num_classes,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            return_quant_tensor=True,
            bias = False
            )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, qnn.QuantConv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        weight_bit_width: int = 8,
        act_bit_width: int = 8,
        new_energy: int = 0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        new_energy = self.new_energy
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # print("Block_0", new_energy)
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,\
                    new_energy=new_energy)
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            # print(new_energy)
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    weight_bit_width=weight_bit_width,
                    act_bit_width=act_bit_width,
                    new_energy=new_energy,
                )
            )

        return layers

    def get_energy_conv(self, x, layer_name, stride, padding):
        
        HM_energy = torch.norm(F.conv2d(input=torch_dec2bin(x / x.scale + x.zero_point),\
                                         weight=torch_dec2bin(layer_name.quant_weight().value / layer_name.quant_weight().scale +\
                                                                layer_name.quant_weight().zero_point), stride=stride, padding =padding), p=1)
        return HM_energy
    
    def get_energy_fc(self, x, layer_name ):
        HM_energy = torch.norm(torch.matmul(torch_dec2bin(x / x.scale + x.zero_point), \
                                             torch_dec2bin(layer_name.quant_weight().value / layer_name.quant_weight().scale + \
                                                           layer_name.quant_weight().zero_point).transpose(1,0)), p=1)

        return HM_energy

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        HM_weight = 0
        HM_activation = 0
        HM_energy = 0
        new_HM_energy = 0
        e = 0

        x = self.quant_input(x)

        HM_activation += torch.sum(torch_dec2bin(x / x.scale + x.zero_point))
        HM_energy += self.get_energy_conv(x, self.conv1, stride=2, padding=3)

        if(self.new_energy):
            e = tile_bitline_current_conv(x, self.conv1, precision=self.weight_bit_width, crossbar_size=256)
            new_HM_energy += e
            # print(e)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       
        x = self.maxpool_relu(x)

        # print(self.layer1_0)

        x, layer1_0_HM_activation, layer1_0_HM_energy = self.layer1_0(x)
        HM_activation += layer1_0_HM_activation
        if(self.new_energy):
            new_HM_energy+=layer1_0_HM_energy
        else:
            HM_energy += layer1_0_HM_energy
        # print("Layer_1_0", HM_energy)
        x, layer1_1_HM_activation, layer1_1_HM_energy = self.layer1_1(x)
        HM_activation += layer1_1_HM_activation
        # HM_energy += layer1_1_HM_energy
        if(self.new_energy):
            new_HM_energy+=layer1_1_HM_energy
        else:
            HM_energy += layer1_1_HM_energy

        x, layer2_0_HM_activation, layer2_0_HM_energy = self.layer2_0(x)
        HM_activation += layer2_0_HM_activation
        # HM_energy += layer2_0_HM_energy
        if(self.new_energy):
            new_HM_energy+=layer2_0_HM_energy
        else:
            HM_energy += layer2_0_HM_energy

        x, layer2_1_HM_activation, layer2_1_HM_energy = self.layer2_1(x)
        HM_activation += layer2_1_HM_activation
        # HM_energy += layer2_1_HM_energy

        if(self.new_energy):
            new_HM_energy+=layer2_1_HM_energy
        else:
            HM_energy += layer2_1_HM_energy

        x, layer3_0_HM_activation, layer3_0_HM_energy = self.layer3_0(x)
        HM_activation += layer3_0_HM_activation
        # HM_energy += layer3_0_HM_energy

        if(self.new_energy):
            new_HM_energy+=layer3_0_HM_energy
        else:
            HM_energy += layer3_0_HM_energy

        x, layer3_1_HM_activation, layer3_1_HM_energy = self.layer3_1(x)
        HM_activation += layer3_1_HM_activation
        # HM_energy += layer3_1_HM_energy

        if(self.new_energy):
            new_HM_energy+=layer3_1_HM_energy
        else:
            HM_energy += layer3_1_HM_energy

        x, layer4_0_HM_activation, layer4_0_HM_energy = self.layer4_0(x)
        HM_activation += layer4_0_HM_activation
        # HM_energy += layer4_0_HM_energy

        if(self.new_energy):
            new_HM_energy+=layer4_0_HM_energy
        else:
            HM_energy += layer4_0_HM_energy

        x, layer4_1_HM_activation, layer4_1_HM_energy = self.layer4_1(x)
        HM_activation += layer4_1_HM_activation
        # HM_energy += layer4_1_HM_energy

        if(self.new_energy):
            new_HM_energy+=layer4_1_HM_energy
        else:
            HM_energy += layer4_1_HM_energy

        x = self.avgpool(x)
        x = self.avgpool_relu(x)
        HM_activation += torch.sum(torch_dec2bin(x / x.scale + x.zero_point))

        x = torch.flatten(x, 1)

        HM_energy += self.get_energy_fc(x, self.fc)
        if(self.new_energy):
            e = tile_bitline_current_fc(x, self.fc, precision=self.weight_bit_width, crossbar_size=256)
            # print(e)
            new_HM_energy += e

        x = self.fc(x)

        if(self.new_energy):
            HM_energy = new_HM_energy

        return x, HM_activation, HM_energy

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    weights: Optional[WeightsEnum],
    progress: bool,
    weight_bit_width: int = 8,
    act_bit_width: int = 8,
    new_energy: int = 0,
    **kwargs: Any,
) -> ResNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = ResNet(block, layers, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,new_energy=new_energy, **kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "min_size": (1, 1),
    "categories": _IMAGENET_CATEGORIES,
}


class ResNet18_Weights(WeightsEnum):
    IMAGENET1K_V1 = Weights(
        url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
        transforms=partial(ImageClassification, crop_size=224),
        meta={
            **_COMMON_META,
            "num_params": 11689512,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_ops": 1.814,
            "_file_size": 44.661,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1



#@register_model()
@handle_legacy_interface(weights=("pretrained", ResNet18_Weights.IMAGENET1K_V1))
def resnet18(*, weights: Optional[ResNet18_Weights] = None, progress: bool = True, weight_bit_width: int = 8, act_bit_width: int = 8, new_energy: int = 0, **kwargs: Any) -> ResNet:
    """ResNet-18 from `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`__.

    Args:
        weights (:class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNet18_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
    """
    weights = ResNet18_Weights.verify(weights)

    return _resnet(BasicBlock, [2, 2, 2, 2], weights, progress, weight_bit_width=weight_bit_width, act_bit_width=act_bit_width,new_energy = new_energy, **kwargs)

