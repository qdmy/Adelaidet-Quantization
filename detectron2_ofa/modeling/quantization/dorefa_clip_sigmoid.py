import torch
import math
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
from detectron2_ofa.layers.wrappers import _NewEmptyTensorOp


def step(x, b):
    y = torch.zeros_like(x)
    mask = torch.ge(x - b, 0.0)
    y[mask] = 1.0
    return y


def step_backward(x, b, T, left_end_point, right_end_point):
    b_buf = x - b
    # b_output = 1 / (1.0 + torch.exp(-b_buf * T))
    # temp = b_output * (1.0 - b_output) * T
    # k = 1 / (right_end_point - left_end_point)
    left_end_point = b - left_end_point
    right_end_point = right_end_point - b
    right_T = T / right_end_point
    left_T = T / left_end_point
    output = x.new_zeros(x.shape)
    output = torch.where(b_buf >= 0, 1 / (1.0 + torch.exp(-b_buf * right_T)), output)
    output = torch.where(b_buf < 0, 1 / (1.0 + torch.exp(-b_buf * left_T)), output)
    output = torch.where(b_buf >= 0, output * (1 - output) * right_T, output)
    output = torch.where(b_buf < 0, output * (1 - output) * left_T, output)
    return output


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, x, b, T, left_end_point, right_end_point):
        self.T = T
        grad = step_backward(x, b, self.T, left_end_point, right_end_point)
        self.save_for_backward(grad)
        return step(x, b)

    @staticmethod
    def backward(self, grad_output):
        grad, = self.saved_tensors
        grad_input = grad * grad_output
        return grad_input, -grad_input, None, None, None


def quantization(x, k, b, T):
    n = 2 ** k - 1
    n = int(n)
    scale = 1 / n

    with torch.no_grad():
        mask = x.new_zeros(x.shape)
        interval_endpoints = []
        interval_endpoints.append(x.new_tensor(0.0))
        for i in range(n - 1):
            interval_endpoint = (b[i] + b[i + 1]) / 2.0
            interval_endpoints.append(interval_endpoint)
            mask = torch.where(x > interval_endpoint, x.new_tensor([i + 1]), mask)
        interval_endpoints.append(x.new_tensor(1.0))
        interval_endpoints = torch.stack(interval_endpoints, dim=0).reshape(-1)

    with torch.no_grad():
        # mask shape: (nelement, 1)
        reshape_mask = mask.reshape(-1, 1).long()
        nelement = reshape_mask.shape[0]
    # expand_b shape: (nelement, n)
    expand_b = b.unsqueeze(0).expand(nelement, n)
    with torch.no_grad():
        # expand_interval_endpoints shape: (nelement, -1)
        expand_interval_endpoints = interval_endpoints.unsqueeze(0).expand(nelement, -1)

    # B shape: (nelement)
    B = torch.gather(expand_b, 1, reshape_mask)
    with torch.no_grad():
        left_end_point = torch.gather(expand_interval_endpoints, 1, reshape_mask)
        right_end_point = torch.gather(expand_interval_endpoints, 1, reshape_mask + 1)
    B = B.reshape(x.shape)
    with torch.no_grad():
        left_end_point = left_end_point.reshape(x.shape)
        right_end_point = right_end_point.reshape(x.shape)
    output = scale * (
        mask + StepFunction.apply(x, B, T, left_end_point, right_end_point)
    )
    return output


def gradient_scale_function(x, scale):
    y_out = x
    y_grad = x * scale
    y = (y_out - y_grad).detach() + y_grad
    return y


def normalization_on_weights(x, clip_value):
    x = x / clip_value
    x = torch.where(x.abs() < 1, x, x.sign())
    return x


def normalization_on_activations(x, clip_value):
    x = F.relu(x)
    x = x / clip_value
    x = torch.where(x < 1, x, x.new_ones(x.shape))
    return x


def quantize_activation(x, k, clip_value, activation_bias, T):
    if k == 32:
        return x
    n = 2 ** k - 1
    grad_scale = math.sqrt(n * x.nelement())
    activation_bias = gradient_scale_function(activation_bias, 1 / grad_scale)
    x = normalization_on_activations(x, clip_value)
    x = quantization(x, k, activation_bias, T)
    x = x * clip_value
    return x


def quantize_weight(x, k, clip_value, weight_bias, T):
    if k == 32:
        return x
    n = 2 ** k - 1
    grad_scale = math.sqrt(n * x.nelement())
    weight_bias = gradient_scale_function(weight_bias, 1 / grad_scale)
    x = normalization_on_weights(x, clip_value)
    x = (x + 1.0) / 2.0
    x = quantization(x, k, weight_bias, T)
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
        use_uniform_quantization=True,
        T=3,
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
        self.weight_n = int(2 ** bits_weights - 1)
        self.activation_n = int(2 ** bits_activations - 1)
        self.use_uniform_quantization = use_uniform_quantization
        self.T = T

        self.weight_level = []
        self.activation_level = []
        self.weight_init_thrs = []
        self.activation_init_thrs = []
        if bits_weights != 32:
            for i in range(int(self.weight_n) + 1):
                self.weight_level.append(float(i) / self.weight_n)
            for i in range(int(self.weight_n)):
                self.weight_init_thrs.append(
                    (self.weight_level[i] + self.weight_level[i + 1]) / 2
                )
        
        if bits_activations != 32:
            for i in range(int(self.activation_n) + 1):
                self.activation_level.append(float(i) / self.activation_n)
            for i in range(int(self.activation_n)):
                self.activation_init_thrs.append(
                    (self.activation_level[i] + self.activation_level[i + 1]) / 2
                )
        
        if use_uniform_quantization:
            self.register_buffer("weight_bias", torch.Tensor(self.weight_init_thrs))
            self.register_buffer("activation_bias", torch.Tensor(self.activation_init_thrs))
        else:
            self.weight_bias = nn.Parameter(torch.Tensor(self.weight_init_thrs))
            self.activation_bias = nn.Parameter(torch.Tensor(self.activation_init_thrs))

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
                x, self.bits_activations, self.activation_clip_value.abs(),
                self.activation_bias.abs(),
                self.T,
            )
            weight_mean = self.weight.data.mean()
            weight_std = self.weight.data.std()
            normalized_weight = self.weight.add(-weight_mean).div(weight_std)
            quantized_weight = quantize_weight(
                normalized_weight,
                self.bits_weights,
                self.weight_clip_value.abs(),
                self.weight_bias.abs(),
                self.T,
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
        s += ", T={}".format(self.T)
        s += ", uniform={}".format(self.use_uniform_quantization)
        s += ", method={}".format("dorefa_clip_rcf_wn_conv_sigmoid")
        return s