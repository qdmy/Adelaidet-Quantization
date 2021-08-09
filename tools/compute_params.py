import copy
import re
import torch
import numpy

from detectron2.config import get_cfg
from detectron2.modeling import build_model

from pytorch_model.resnet import ResNet

config_file = "/home/liujing/Codes/detections/Adelaidet-Quantization/configs/FCOS-COCO-Detection/fcos_R_18_1x-Full_SyncBN_dorefa_clip.yaml"

cfg = get_cfg()
cfg.merge_from_file(config_file)
detectron_model = build_model(cfg)
bn_modules = []
for head in ['cls', 'bbox']:
    for i in range(4):
        bn_modules.append(getattr(detectron_model.proposal_generator.fcos_head, "{}_norm{}".format(head, i)))
# print(bn_modules)

import operator
import functools
param_count = sum([functools.reduce(operator.mul, i.size(), 1) for i in detectron_model.parameters()])
bm_param_count = 0
for bn_module in bn_modules:
    bm_param_count += sum([functools.reduce(operator.mul, i.size(), 1) for i in bn_module.parameters()])
# print(param_count)
print('Total param: {}M'.format(param_count / 1e6))
print('BN param: {}K'.format(bm_param_count / 1e3))
print('Percentage: {}'.format(bm_param_count / param_count))