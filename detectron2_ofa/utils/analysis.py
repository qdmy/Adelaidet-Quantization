# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-

import logging
import typing
from collections import Counter, OrderedDict, defaultdict

import torch
from torch import nn

from detectron2_ofa.structures import BitMasks, Boxes, ImageList, Instances
from detectron2_ofa.utils.model_analysis import ModelAnalyse
from fvcore.nn import (activation_count, flop_count, parameter_count,
                       parameter_count_table)
from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS
from fvcore.nn.jit_handles import (addmm_flop_jit, conv_flop_jit,
                                   einsum_flop_jit, matmul_flop_jit)

from .logger import log_first_n

__all__ = [
    "activation_count_operators",
    "flop_count_operators",
    "parameter_count_table",
    "parameter_count",
]

FLOPS_MODE = "flops"
ACTIVATIONS_MODE = "activations"


# some extra ops to ignore from counting.
_IGNORED_OPS = [
    "aten::batch_norm",
    "aten::div",
    "aten::div_",
    "aten::meshgrid",
    "aten::rsub",
    "aten::sub",
    "aten::relu_",
    "aten::add_",
    "aten::mul",
    "aten::add",
    "aten::relu",
    "aten::sigmoid",
    "aten::sigmoid_",
    "aten::sort",
    "aten::exp",
    "aten::mul_",
    "aten::max_pool2d",
    "aten::constant_pad_nd",
    "aten::sqrt",
    "aten::softmax",
    "aten::log2",
    "aten::nonzero_numpy",
    "prim::PythonOp",
    "torchvision::nms",
]


def flop_count_operators(
    model: nn.Module, inputs: list, **kwargs
) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level flops counting using jit.
    This is a wrapper of fvcore.nn.flop_count, that supports standard detection models
    in detectron2_ofa.
    Note:
        The function runs the input through the model to compute flops.
        The flops of a detection model is often input-dependent, for example,
        the flops of box & mask head depends on the number of proposals &
        the number of detected objects.
        Therefore, the flops counting using a single input may not accurately
        reflect the computation cost of a model.
    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
    """
    return _wrapper_count_operators(model=model, inputs=inputs, mode=FLOPS_MODE, **kwargs)


def activation_count_operators(
    model: nn.Module, inputs: list, **kwargs
) -> typing.DefaultDict[str, float]:
    """
    Implement operator-level activations counting using jit.
    This is a wrapper of fvcore.nn.activation_count, that supports standard detection models
    in detectron2_ofa.
    Note:
        The function runs the input through the model to compute activations.
        The activations of a detection model is often input-dependent, for example,
        the activations of box & mask head depends on the number of proposals &
        the number of detected objects.
    Args:
        model: a detectron2 model that takes `list[dict]` as input.
        inputs (list[dict]): inputs to model, in detectron2's standard format.
    """
    return _wrapper_count_operators(model=model, inputs=inputs, mode=ACTIVATIONS_MODE, **kwargs)


def _flatten_to_tuple(outputs):
    result = []
    if isinstance(outputs, torch.Tensor):
        result.append(outputs)
    elif isinstance(outputs, (list, tuple)):
        for v in outputs:
            result.extend(_flatten_to_tuple(v))
    elif isinstance(outputs, dict):
        for _, v in outputs.items():
            result.extend(_flatten_to_tuple(v))
    elif isinstance(outputs, Instances):
        result.extend(_flatten_to_tuple(outputs.get_fields()))
    elif isinstance(outputs, (Boxes, BitMasks, ImageList)):
        result.append(outputs.tensor)
    else:
        log_first_n(
            logging.WARN,
            f"Output of type {type(outputs)} not included in flops/activations count.",
            n=10,
        )
    return tuple(result)


def _wrapper_count_operators(
    model: nn.Module, inputs: list, mode: str, **kwargs
) -> typing.DefaultDict[str, float]:

    # ignore some ops
    supported_ops = {k: lambda *args, **kwargs: {} for k in _IGNORED_OPS}
    supported_ops.update(kwargs.pop("supported_ops", {}))
    kwargs["supported_ops"] = supported_ops

    assert len(inputs) == 1, "Please use batch size=1"
    tensor_input = inputs[0]["image"]

    class WrapModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            if isinstance(
                model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
            ):
                self.model = model.module
            else:
                self.model = model

        def forward(self, image):
            # jit requires the input/output to be Tensors
            inputs = [{"image": image[0]}]
            outputs = self.model.forward(inputs)
            # Only the subgraph that computes the returned tuple of tensor will be
            # counted. So we flatten everything we found to tuple of tensors.
            return _flatten_to_tuple(outputs)

    old_train = model.training
    with torch.no_grad():
        if mode == FLOPS_MODE:
            ret = flop_count(WrapModel(model).train(False), (tensor_input,), **kwargs)
        elif mode == ACTIVATIONS_MODE:
            ret = activation_count(WrapModel(model).train(False), (tensor_input,), **kwargs)
        else:
            raise NotImplementedError("Count for mode {} is not supported yet.".format(mode))
    # compatible with change in fvcore
    if isinstance(ret, tuple):
        ret = ret[0]
    model.train(old_train)
    return ret

def flop_count(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    supported_ops: typing.Union[typing.Dict[str, typing.Callable], None] = None,
) -> typing.Tuple[typing.DefaultDict[str, float], typing.Counter[str]]:
    """
    Given a model and an input to the model, compute the Gflops of the given
    model.

    Args:
        model (nn.Module): The model to compute flop counts.
        inputs (tuple): Inputs that are passed to `model` to count flops.
            Inputs need to be in a tuple.
        supported_ops (dict(str,Callable) or None) : provide additional
            handlers for extra ops, or overwrite the existing handlers for
            convolution and matmul and einsum. The key is operator name and the value
            is a function that takes (inputs, outputs) of the op. We count
            one Multiply-Add as one FLOP.

    Returns:
        tuple[defaultdict, Counter]: A dictionary that records the number of
            gflops for each operation and a Counter that records the number of
            skipped operations.
    """
    assert isinstance(inputs, tuple), "Inputs need to be in a tuple."
    supported_ops = {**_DEFAULT_SUPPORTED_OPS, **(supported_ops or {})}

    # Run flop count.
    # total_flop_counter, skipped_ops = get_jit_model_analysis(
    #     model, inputs, supported_ops
    # )
    model_analyse = ModelAnalyse(model)
    model_analyse.flops_compute(inputs)

    # Log for skipped operations.
    logger = logging.getLogger(__name__)
    # if len(skipped_ops) > 0:
    #     for op, freq in skipped_ops.items():
    #         logger.warning("Skipped operation {} {} time(s)".format(op, freq))

    # # Convert flop count to gigaflops.
    # final_count = defaultdict(float)
    # for op in total_flop_counter:
    #     final_count[op] = total_flop_counter[op] / 1e9

    # return final_count, skipped_ops

def get_jit_model_analysis(
    model: nn.Module,
    inputs: typing.Tuple[object, ...],
    ops_handles: typing.Dict[str, typing.Callable],
) -> typing.Tuple[typing.Counter[str], typing.Counter[str]]:
    """
    Given a model, the inputs and the handles for each operation, return the
    results for the model analysis.

    Args:
        model (nn.Module): The model for torch script to trace.
        inputs (tuple): Inputs that are passed to `model` to trace. Inputs need
            to be in a tuple.
        ops_handles (typing.Dict[str, typing.Callable]): A dictionary of handles
            for model analysis.

    Returns:
        typing.Tuple[typing.Counter[str], typing.Counter[str]]: A counter that
            contains the results of per operation analysis of the model and a
            Counter of ignored operations.
    """
    # Torch script does not support parallel torch models.
    if isinstance(
        model, (nn.parallel.distributed.DistributedDataParallel, nn.DataParallel)
    ):
        model = model.module  # pyre-ignore

    # Compatibility with torch.jit.
    if hasattr(torch.jit, "get_trace_graph"):
        trace, _ = torch.jit.get_trace_graph(model, inputs)
        trace_nodes = trace.graph().nodes()
    else:
        trace, _ = torch.jit._get_trace_graph(model, inputs)
        trace_nodes = trace.nodes()

    skipped_ops = Counter()
    total_count = Counter()

    for node in trace_nodes:
        kind = node.kind()
        print(kind)
        if kind not in ops_handles.keys():
            # If the operation is not in _IGNORED_OPS, count skipped operations.
            if kind not in _IGNORED_OPS:
                skipped_ops[kind] += 1
            continue

        handle_count = ops_handles.get(kind, None)
        if handle_count is None:
            continue
        # pyre-ignore
        inputs, outputs = list(node.inputs()), list(node.outputs())
        op_count = handle_count(inputs, outputs)
        total_count += op_count
    return total_count, skipped_ops
