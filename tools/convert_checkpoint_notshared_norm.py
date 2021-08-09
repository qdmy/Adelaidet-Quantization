import torch

pretrained_model_path = "/userhome/liujing/models/chenpeng/R_18_1x-FPN_BN/model_final.pth"
save_model_path = "/userhome/liujing/models/chenpeng/R_18_1x-FPN_BN/model_final_notshared_norm.pth"


# load pre-trained model into pytorch_model
checkpoint_param = torch.load(pretrained_model_path, map_location='cpu')
prefixes = ["proposal_generator.fcos_head.cls_tower", "proposal_generator.fcos_head.bbox_tower"]
output_prefixes = ["proposal_generator.fcos_head.cls_norm", "proposal_generator.fcos_head.bbox_norm"]
# only for GN

j = 0
for i in range(0, 10, 3):
    for prefix, output_prefix in zip(prefixes, output_prefixes):
        ker_norm_weight_name = '{}.{}.weight'.format(prefix, i + 1)
        key_norm_bias_name = '{}.{}.bias'.format(prefix, i + 1)
        key_names = [ker_norm_weight_name, key_norm_bias_name]
        for key_name in key_names:
            value = checkpoint_param["model"].pop(key_name)
            for k in range(5):
                if 'weight' in key_name:
                    output_key_name = '{}{}.{}.weight'.format(output_prefix, j, k)
                else:
                    output_key_name = '{}{}.{}.bias'.format(output_prefix, j, k)
                checkpoint_param["model"][output_key_name] = value
    j += 1
torch.save(checkpoint_param, save_model_path)
