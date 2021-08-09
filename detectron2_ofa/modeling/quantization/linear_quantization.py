import torch
import numpy as np
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from detectron2_ofa.layers.wrappers import Conv2d


def linear_quantize(input, scale, zero_point):
    return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point):
    return (input + zero_point) / scale


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point):
        output = linear_quantize(input, scale, zero_point)
        output = linear_dequantize(output, scale, zero_point)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None


def get_tensor_min_max(t, per_dim=None):
    if per_dim is None:
        return t.min(), t.max()
    if per_dim >= t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0], tv.max(dim=-1)[0]


def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max):
    n = 2.0 ** num_bits - 1

    # Make sure 0 is in the range
    sat_min = torch.min(saturation_min, torch.zeros_like(saturation_min))
    sat_max = torch.max(saturation_max, torch.zeros_like(saturation_max))

    diff = sat_max - sat_min
    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    diff[diff == 0] = n

    scale = n / diff
    zero_point = scale * sat_min
    zero_point = zero_point.round()
    return scale, zero_point


def clamp(input, min, max):
    return torch.clamp(input, min, max)


def all_max(input):
    input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
    # Use allgather instead of allreduce since I don't trust in-place operations ..
    dist.all_gather(input_list, input, async_op=False)
    inputs = torch.stack(input_list, dim=0)
    return torch.max(inputs, dim=0)[0]


class ActivationStatisticsQuantization(nn.Module):
    def __init__(self, num_bits=8, percentile=0.999, get_statistics=True):
        super(ActivationStatisticsQuantization, self).__init__()

        self.get_statistics = get_statistics
        self.register_buffer('num_bits', torch.tensor([num_bits]))
        self.register_buffer('percentile', torch.tensor([percentile]))

        # We track activations ranges with exponential moving average, as proposed by Jacob et al., 2017
        # https://arxiv.org/abs/1712.05877
        # We perform bias correction on the EMA, so we keep both unbiased and biased values and the iterations count
        # For a simple discussion of this see here:
        # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
        # self.register_buffer('tracked_min_biased', torch.zeros(1))
        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_min_avg', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('tracked_max_avg', torch.zeros(1))
        self.register_buffer('iter_count', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def forward(self, input):
        if self.num_bits == 32.0:
            return input
        elif self.get_statistics:
            with torch.no_grad():
                min = input.new_tensor(np.percentile(input.cpu().numpy(), (1 - self.percentile).item() * 100))
                max = input.new_tensor(np.percentile(input.cpu().numpy(), self.percentile.item() * 100))
            self.iter_count += 1

            min = min.unsqueeze(0)
            max = max.unsqueeze(0)
            vec = torch.cat([min, max], dim=0)
            vec = all_max(vec)

            min, max = torch.split(vec, 1)

            if self.iter_count == 1:
                self.tracked_min = min
                self.tracked_max = max
                self.tracked_min_avg = min
                self.tracked_max_avg = max
            else:
                self.tracked_min = self.tracked_min.data + min.data
                self.tracked_max = self.tracked_max.data + max.data
                self.tracked_min_avg = self.tracked_min.data / self.iter_count
                self.tracked_max_avg = self.tracked_max.data / self.iter_count
                self.scale.data, self.zero_point.data = asymmetric_linear_quantization_params(self.num_bits,
                                                                                              self.tracked_min_avg,
                                                                                              self.tracked_max_avg)
        else:
            actual_min, actual_max = self.tracked_min_avg, self.tracked_max_avg
            # print("Min: {}, Max: {}".format(actual_min, actual_max))
            # print(actual_min)
            # print(actual_max)
            # if self.training:
            #     self.scale.data, self.zero_point.data = asymmetric_linear_quantization_params(self.num_bits,
            #                                                                                   self.tracked_min,
            #                                                                                   self.tracked_max)

            qinput = clamp(input, actual_min.item(), actual_max.item())
            qinput = LinearQuantizeSTE.apply(qinput, self.scale, self.zero_point)
            return qinput

        return input

    def extra_repr(self):
        return 'num_bits={0})'.format(self.num_bits)


class QConv2d(Conv2d):
    """
    custom convolutional layers for quantization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, norm=None, activation=None,
                 bits_weights=32.0, bits_activations=32.0, percentile=0.999, get_statistics=True, merge_bn=False):
        super(QConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias, norm=norm, activation=activation)
        self.get_statistics = get_statistics
        self.merge_bn = merge_bn
        self.register_buffer('iter_count', torch.zeros(1))
        self.register_buffer('bits_weights', torch.FloatTensor([bits_weights]))
        self.register_buffer('bits_activations', torch.FloatTensor([bits_activations]))
        self.fake_q = ActivationStatisticsQuantization(bits_activations, percentile, get_statistics=get_statistics)

    def fold_bn(self, mean, std):
        gamma_ = self.norm.weight / std
        weight = self.weight * gamma_.view(self.out_channels, 1, 1, 1)
        if self.bias is not None:
            bias = gamma_ * self.bias - gamma_ * mean + self.norm.bias
        else:
            bias = self.norm.bias - gamma_ * mean
            
        return weight, bias

    def forward(self, input):
        # quantize input
        quantized_input = self.fake_q(input)

        # quantize weight
        if self.merge_bn:
            if self.norm is not None:
                std = torch.sqrt(self.norm.running_var + self.norm.eps)
                weight, bias = self.fold_bn(self.norm.running_mean, std)

                if self.bits_weights == 32.0:
                    quantized_weight = weight
                    quantized_bias = bias
                elif self.get_statistics:
                    self.iter_count += 1
                    quantized_weight = weight
                    quantized_bias = bias
                    if self.iter_count > 20:
                        self.get_statistics = False
                        self.fake_q.get_statistics = False
                else:
                    # get channel-wise min and max value
                    weight_min, weight_max = get_tensor_min_max(weight, per_dim=0)
                    with torch.no_grad():
                        scale, zero_point = asymmetric_linear_quantization_params(self.bits_weights, weight_min, weight_max)

                    # Reshape scale and zero_points so they can be broadcast properly with the weight tensor
                    dims = [scale.shape[0]] + [1] * (weight.dim() - 1)
                    scale = scale.view(dims)
                    zero_point = zero_point.view(dims)

                    quantized_weight = LinearQuantizeSTE.apply(weight, scale, zero_point)
                    quantized_bias = bias

                output = F.conv2d(quantized_input, quantized_weight, quantized_bias, self.stride,
                                self.padding, self.dilation, self.groups)
            else:
                if self.bits_weights == 32.0:
                    quantized_weight = self.weight
                elif self.get_statistics:
                    self.iter_count += 1
                    quantized_weight = self.weight
                    if self.iter_count > 20:
                        self.get_statistics = False
                        self.fake_q.get_statistics = False
                else:
                    # get channel-wise min and max value
                    weight_min, weight_max = get_tensor_min_max(self.weight, per_dim=0)
                    with torch.no_grad():
                        scale, zero_point = asymmetric_linear_quantization_params(self.bits_weights, weight_min, weight_max)

                    # Reshape scale and zero_points so they can be broadcast properly with the weight tensor
                    dims = [scale.shape[0]] + [1] * (self.weight.dim() - 1)
                    scale = scale.view(dims)
                    zero_point = zero_point.view(dims)

                    quantized_weight = LinearQuantizeSTE.apply(self.weight, scale, zero_point)

                output = F.conv2d(quantized_input, quantized_weight, self.bias, self.stride,
                                self.padding, self.dilation, self.groups)
        else:
            if self.bits_weights == 32.0:
                quantized_weight = self.weight
            elif self.get_statistics:
                self.iter_count += 1
                quantized_weight = self.weight
                if self.iter_count > 20:
                    self.get_statistics = False
                    self.fake_q.get_statistics = False
            else:
                # get channel-wise min and max value
                weight_min, weight_max = get_tensor_min_max(self.weight, per_dim=0)
                with torch.no_grad():
                    scale, zero_point = asymmetric_linear_quantization_params(self.bits_weights, weight_min, weight_max)

                # Reshape scale and zero_points so they can be broadcast properly with the weight tensor
                dims = [scale.shape[0]] + [1] * (self.weight.dim() - 1)
                scale = scale.view(dims)
                zero_point = zero_point.view(dims)

                quantized_weight = LinearQuantizeSTE.apply(self.weight, scale, zero_point)

            output = F.conv2d(quantized_input, quantized_weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
            
            if self.norm is not None:
                output = self.norm(output)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def extra_repr(self):
        s = super().extra_repr()
        s += ', bits_weights={}'.format(self.bits_weights.item())
        s += ', bits_activations={}'.format(self.bits_activations.item())
        s += ', merge_bn={}'.format(self.merge_bn)
        s += ', method={}'.format('linear')
        return s
