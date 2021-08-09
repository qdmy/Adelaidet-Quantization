import copy
import re
import torch

from detectron2.config import get_cfg
from detectron2.modeling import build_model

from pytorch_model.resnet import ResNet


def convert_basic_pytorch_names(original_keys):
    layer_keys = copy.deepcopy(original_keys)
    layer_keys = [k.replace("layer1", "res2") for k in layer_keys]
    layer_keys = [k.replace("layer2", "res3") for k in layer_keys]
    layer_keys = [k.replace("layer3", "res4") for k in layer_keys]
    layer_keys = [k.replace("layer4", "res5") for k in layer_keys]
    # layer_keys = [k.replace("weight_clip_value", "weight_clip_val") for k in layer_keys]
    # layer_keys = [k.replace("activation_clip_value", "activation.clip_val") for k in layer_keys]

    layer_keys = [k.replace("bn1", "conv1.norm") for k in layer_keys]
    layer_keys = [k.replace("bn2", "conv2.norm") for k in layer_keys]
    layer_keys = [k.replace("bn3", "conv3.norm") for k in layer_keys]

    layer_keys = [re.sub("^conv", "stem.conv", k) for k in layer_keys]
    layer_keys = [re.sub("downsample.0", "shortcut", k) for k in layer_keys]
    layer_keys = [re.sub("downsample.1", "shortcut.norm", k) for k in layer_keys]
    # layer_keys = [re.sub("^res", "backbone.bottom_up.res", k) for k in layer_keys]

    layer_keys = ["backbone.bottom_up.{}".format(k) for k in layer_keys]
    return layer_keys

def forworad_pytorch_model(pytorch_model, input):
    x = pytorch_model.conv1(input)
    x = pytorch_model.bn1(x)
    x = pytorch_model.relu(x)
    x = pytorch_model.maxpool(x)

    x = pytorch_model.layer1(x)
    x = pytorch_model.layer2(x)
    x = pytorch_model.layer3(x)
    x = pytorch_model.layer4(x)
    return x

def forward_detectron_model(detectron_model, input):
    x = detectron_model.backbone.bottom_up.stem(input)
    for stage, name in detectron_model.backbone.bottom_up.stages_and_names:
        x = stage(x)
    return x

depth = 50
pretrained_model_path = "/projects/dl65/liujing/models/resnet50_official.pth"
save_model_path = "/projects/dl65/liujing/models/resnet50_detectron.pth"
config_file = "/projects/dl65/liujing/Adelaidet-Quantization/configs/COCO-Detection/retinanet_R_50_FPN_1x-FPN_BN-Head_GN.yaml"

# construct pytorch model and detectron model
pytorch_model = ResNet(depth).cuda()

cfg = get_cfg()
cfg.merge_from_file(config_file)
detectron_model = build_model(cfg)
print(detectron_model)

# load pre-trained model into pytorch_model
checkpoint_param = torch.load(pretrained_model_path)
pytorch_model.load_state_dict(checkpoint_param, strict=False)

# get state dict
# pytorch_model_state_dict = pytorch_model.state_dict()
pytorch_model_state_dict = checkpoint_param
pytorch_model_keys = sorted(list(pytorch_model_state_dict.keys()))
# print("Pytorch Model")
# print(pytorch_model_keys)

detectron_model_state_dict = detectron_model.state_dict()
detectron_model_keys = sorted(list(detectron_model_state_dict.keys()))
# print("Detectron2 Model")
# print(detectron_model_keys)

converted_keys = convert_basic_pytorch_names(pytorch_model_keys)
# print("Converted kyes")
# print(converted_keys)

new_weights = {}
new_keys_to_original_keys = {}
for orig, renamed in zip(pytorch_model_keys, converted_keys):
    new_keys_to_original_keys[renamed] = orig
    new_weights[renamed] = pytorch_model_state_dict[orig]

incompatible = detectron_model.load_state_dict(new_weights, strict=False)
if incompatible.missing_keys:
    print('Missing Keys')
    print(incompatible.missing_keys)
if incompatible.unexpected_keys:
    print('Incompatible Keys')
    print(incompatible.unexpected_keys)


# test_input = torch.randn(1, 3, 224, 224).cuda()
# pytorch_model.eval()
# detectron_model.eval()
# original_results = forworad_pytorch_model(pytorch_model, test_input)
# detectron_results = forward_detectron_model(detectron_model, test_input)
# print((original_results - detectron_results).max())

# print(original_results)
# print(detectron_results)
# print(pytorch_model.conv1.weight)
# print(detectron_model.backbone.bottom_up.stem.conv1.weight)

torch.save(new_weights, save_model_path)
# for name, module in pytorch_model.named_children():
#     print(name)