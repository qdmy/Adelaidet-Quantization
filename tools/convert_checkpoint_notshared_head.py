import torch

pretrained_model_path = "/userhome/liujing/models/chenpeng/R_18_1x-FPN_BN/model_final.pth"
save_model_path = "/userhome/liujing/models/chenpeng/R_18_1x-FPN_BN/model_final_notshared_head.pth"


# load pre-trained model into pytorch_model
checkpoint_param = torch.load(pretrained_model_path, map_location='cpu')
prefixes = ["proposal_generator.fcos_head.cls_tower", "proposal_generator.fcos_head.bbox_tower"]
output_prefixes = ["proposal_generator.fcos_head.head_cls_tower", "proposal_generator.fcos_head.head_bbox_tower"]
# only for GN

for i in range(0, 10, 3):
    for prefix, output_prefix in zip(prefixes, output_prefixes):
        key_weight_name = '{}.{}.weight'.format(prefix, i)
        # key_bias_name = '{}.{}.bias'.format(prefix, i)
        ker_norm_weight_name = '{}.{}.weight'.format(prefix, i + 1)
        key_norm_bias_name = '{}.{}.bias'.format(prefix, i + 1)
        key_names = [key_weight_name, ker_norm_weight_name, key_norm_bias_name]
        for k, key_name in enumerate(key_names):
            value = checkpoint_param["model"].pop(key_name)
            for j in range(5):
                if 'weight' in key_name:
                    if k < 1:
                        output_key_weight_name = '{}.{}.{}.weight'.format(output_prefix, j, i)
                    else:
                        output_key_weight_name = '{}.{}.{}.weight'.format(output_prefix, j, i + 1)
                else:
                    output_key_weight_name = '{}.{}.{}.bias'.format(output_prefix, j, i + 1)
                checkpoint_param["model"][output_key_weight_name] = value
torch.save(checkpoint_param, save_model_path)
