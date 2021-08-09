# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2_ofa.utils.registry import Registry
from third_party.quantization.convert_to_quantization import convert2quantization

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    convert2quantization(model, cfg)
    return model
