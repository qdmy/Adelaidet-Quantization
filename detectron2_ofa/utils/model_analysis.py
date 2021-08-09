import numpy as np
import torch.nn as nn
from prettytable import PrettyTable

from collections import OrderedDict

__all__ = ["ModelAnalyse"]


class ModelAnalyse(object):
    def __init__(self, model):
        self.model = model
        self.flops = OrderedDict()
        self.madds = OrderedDict()

    def params_count(self):
        params_num_list = []

        output = PrettyTable()
        output.field_names = ["Param name", "Shape", "Dim"]

        print("------------------------number of parameters------------------------\n")
        for name, param in self.model.named_parameters():
            param_num = param.numel()
            param_shape = [shape for shape in param.shape]
            params_num_list.append(param_num)
            output.add_row([name, param_shape, param_num])
        print(output)

        params_num_list = np.array(params_num_list)
        params_num = params_num_list.sum()
        print("|===>Number of parameters is: {:}, {:f} M".format(params_num, params_num / 1e6))
        return params_num

    def _flops_conv_hook(self, layer, x, out):
        # https://www.zhihu.com/question/65305385

        # compute number of floating point operations
        in_channels = layer.in_channels
        groups = layer.groups
        channels_per_filter = in_channels // groups
        if layer.bias is not None:
            layer_flops = out.size(2) * out.size(3) * \
                          (2. * channels_per_filter * layer.weight.size(2) * layer.weight.size(3)) \
                          * layer.weight.size(0)
        else:
            layer_flops = out.size(2) * out.size(3) * \
                          (2. * channels_per_filter * layer.weight.size(2) * layer.weight.size(3) - 1.) \
                          * layer.weight.size(0)

        layer_name = layer.layer_name
        print(self.flops)
        if layer_name in self.flops:
            self.flops[layer_name] += layer_flops
        else:
            self.flops[layer_name] = layer_flops
        # if we only care about multipy operation, use followiGng equation instead
        """
        layer_flops = out.size(2)*out.size(3)*layer.weight.size(1)*layer.weight.size(2)*layer.weight.size(0)
        """

    def _flops_linear_hook(self, layer, x, out):
        # compute number floating point operations
        if layer.bias is not None:
            layer_flops = (2 * layer.weight.size(1)) * layer.weight.size(0)
        else:
            layer_flops = (2 * layer.weight.size(1) - 1) * layer.weight.size(0)
        # if we only care about multipy operation, use following equation instead
        """
        layer_flops = layer.weight.size(1)*layer.weight.size(0)
        """

        layer_name = layer.layer_name
        if layer_name in self.flops:
            self.flops[layer_name] += layer_flops
        else:
            self.flops[layer_name] = layer_flops

    def _madds_conv_hook(self, layer, x, out):
        input = x[0]
        batch_size = input.shape[0]
        output_height, output_width = out.shape[2:]

        kernel_height, kernel_width = layer.kernel_size
        if hasattr(layer, 'd'):
            in_channels = layer.d.sum().item()
        else:
            in_channels = layer.in_channels
        out_channels = layer.out_channels
        groups = layer.groups

        filters_per_channel = out_channels // groups
        conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

        active_elements_count = batch_size * output_height * output_width

        overall_conv_flops = conv_per_position_flops * active_elements_count

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out_channels * active_elements_count

        overall_flops = overall_conv_flops + bias_flops
        layer_name = layer.layer_name
        if layer_name in self.madds:
            self.madds[layer_name] += overall_flops
        else:
            self.madds[layer_name] = overall_flops

    def _madds_linear_hook(self, layer, x, out):
        # compute number of multiply-add
        # layer_madds = layer.weight.size(0) * layer.weight.size(1)
        # if layer.bias is not None:
        #     layer_madds += layer.weight.size(0)
        input = x[0]
        batch_size = input.shape[0]
        overall_flops = int(batch_size * input.shape[1] * out.shape[1])

        bias_flops = 0
        if layer.bias is not None:
            bias_flops = out.shape[1]
        overall_flops = overall_flops + bias_flops
        layer_name = layer.layer_name
        if layer_name in self.madds:
            self.madds[layer_name] += overall_flops
        else:
            self.madds[layer_name] = overall_flops

    def madds_compute(self, x):
        """
        Compute number of multiply-adds of the model
        """

        hook_list = []
        self.madds = OrderedDict()
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._madds_conv_hook))
                layer.layer_name = layer_name
                # self.layer_names.append(layer_name)
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._madds_linear_hook))
                layer.layer_name = layer_name
                # self.layer_names.append(layer_name)
        # run forward for computing FLOPs
        self.model.eval()
        self.model(x)

        madds_sum = 0.0
        for k, v in self.madds.items():
            madds_sum += v

        output = PrettyTable()
        output.field_names = ["Layer", "Madds", "Percentage"]

        print("------------------------Madds------------------------\n")
        for k, v in self.madds.items():
            output.add_row([k, v, v / madds_sum])
        print(output)
        repo_str = "|===>Total MAdds: {:e}".format(madds_sum)
        print(repo_str)

        for hook in hook_list:
            hook.remove()

    def flops_compute(self, x):
        """
        Compute number of flops of the model
        """

        hook_list = []
        self.flops = OrderedDict()
        for layer_name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                hook_list.append(layer.register_forward_hook(self._flops_conv_hook))
                layer.layer_name = layer_name
                # self.layer_names.append(layer_name)
            elif isinstance(layer, nn.Linear):
                hook_list.append(layer.register_forward_hook(self._flops_linear_hook))
                layer.layer_name = layer_name
                # self.layer_names.append(layer_name)

        # run forward for computing FLOPs
        # self.model.eval()
        self.model(x)

        flops_sum = 0.0
        for k, v in self.flops.items():
            flops_sum += v

        output = PrettyTable()
        output.field_names = ["Layer", "FLOPs", "Percentage"]

        print("------------------------FLOPs------------------------\n")
        for k, v in self.flops.items():
            output.add_row([k, v, v / flops_sum])
        print(output)
        repo_str = "|===>Total FLOPs: {:e} FLOPs, {:f} MFLOPs".format(flops_sum, flops_sum / 1e6)
        print(repo_str)

        for hook in hook_list:
            hook.remove()
