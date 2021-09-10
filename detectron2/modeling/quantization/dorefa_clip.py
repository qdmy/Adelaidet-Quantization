import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from detectron2.layers.wrappers import _NewEmptyTensorOp


def quantization(x, k):
    n = 2 ** k - 1
    return RoundFunction.apply(x, n)


def normalization_on_weights(x, clip_value):
    x = x / clip_value
    x = torch.where(x.abs() < 1, x, x.sign())
    return x


def normalization_on_activations(x, clip_value):
    x = F.relu(x)
    x = x / clip_value
    x = torch.where(x < 1, x, x.new_ones(x.shape))
    return x


def quantize_activation(x, k, clip_value):
    if k == 32:
        return x
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k)
    x = x * clip_value
    return x


def quantize_weight(x, k, clip_value):
    if k == 32:
        return x
    x = normalization_on_weights(x, clip_value)
    x = (x + 1.0) / 2.0
    x = quantization(x, k)
    x = x * 2.0 - 1.0
    x = x * clip_value
    return x


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x, n):
        return torch.round(x * n) / n

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class QConv2d(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bits_weights=32,
        bits_activations=32,
        **kwargs
    ):
        super(QConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        # self.eps = 1e-5
        self.norm = kwargs.pop("norm", None)
        self.activation = kwargs.pop("activation", None)
        self.weight_clip_value = nn.Parameter(torch.Tensor([1]))
        self.activation_clip_value = nn.Parameter(torch.Tensor([1]))
        self.target_bits_weights = bits_weights
        self.target_bits_activations = bits_activations
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty
        else:
            quantized_input = quantize_activation(
                x, self.bits_activations, self.activation_clip_value
            )
            weight_mean = self.weight.data.mean()
            weight_std = self.weight.data.std()
            normalized_weight = self.weight.add(-weight_mean).div(weight_std)
            quantized_weight = quantize_weight(
                normalized_weight, self.bits_weights, self.weight_clip_value
            )
            output = F.conv2d(
                quantized_input,
                quantized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

            if self.norm is not None:
                output = self.norm(output)
            if self.activation is not None:
                output = self.activation(output)
            return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_clip_rcf_wn_conv")
        return s


class QConv2dMultiLevel(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        bits_weights=32,
        bits_activations=32,
        level=5,
        **kwargs
    ):
        super(QConv2dMultiLevel, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        # self.eps = 1e-5
        self.norm = kwargs.pop("norm", None)
        self.activation = kwargs.pop("activation", None)
        self.weight_clip_value_list = nn.ParameterList(nn.Parameter(torch.Tensor([1])) for i in range(level))
        self.activation_clip_value_list = nn.ParameterList(nn.Parameter(torch.Tensor([1])) for i in range(level))
        self.weight_clip_value = self.weight_clip_value_list[0]
        self.activation_clip_value = self.activation_clip_value_list[0]
        self.target_bits_weights = bits_weights
        self.target_bits_activations = bits_activations
        self.bits_weights = bits_weights
        self.bits_activations = bits_activations

    def forward(self, x):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            empty = _NewEmptyTensorOp.apply(x, output_shape)
            if self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

                # This is to make DDP happy.
                # DDP expects all workers to have gradient w.r.t the same set of parameters.
                _dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + _dummy
            else:
                return empty
        else:
            quantized_input = quantize_activation(
                x, self.bits_activations, self.activation_clip_value
            )
            weight_mean = self.weight.data.mean()
            weight_std = self.weight.data.std()
            normalized_weight = self.weight.add(-weight_mean).div(weight_std)
            quantized_weight = quantize_weight(
                normalized_weight, self.bits_weights, self.weight_clip_value
            )
            output = F.conv2d(
                quantized_input,
                quantized_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

            if self.norm is not None:
                output = self.norm(output)
            if self.activation is not None:
                output = self.activation(output)
            return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ", bits_weights={}".format(self.bits_weights)
        s += ", bits_activations={}".format(self.bits_activations)
        s += ", method={}".format("dorefa_clip_rcf_wn_conv_multi_level")
        return s