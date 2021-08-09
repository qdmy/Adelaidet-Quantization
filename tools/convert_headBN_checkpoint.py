import torch

pretrained_model_path = "/home/liujing/Models/chenpeng/R_18_1x-Full_BN/model_final.pth"
save_model_path = "/home/liujing/Models/chenpeng/R_18_1x-Full_BN/model_final_multilevel.pth"


# load pre-trained model into pytorch_model
checkpoint_param = torch.load(pretrained_model_path)
prefixes = ["proposal_generator.fcos_head.cls_tower", "proposal_generator.fcos_head.bbox_tower"]
norm_prefiexes = ["proposal_generator.fcos_head.cls_norm", "proposal_generator.fcos_head.bbox_norm"]
output_prefixes = ["proposal_generator.fcos_head.head_cls_tower", "proposal_generator.fcos_head.head_bbox_tower"]
# only for GN

for i in range(0, 10, 3):
    for prefix, output_prefix in zip(prefixes, output_prefixes):
        key_weight_name = '{}.{}.weight'.format(prefix, i)
        value = checkpoint_param["model"].pop(key_weight_name)

        for j in range(5):
            output_key_weight_name = '{}.{}.{}.weight'.format(output_prefix, j, i)
            checkpoint_param["model"][output_key_weight_name] = value

for prefix, output_prefix in zip(norm_prefiexes, output_prefixes):
    key_list = ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']
    for i in range(4):
        for j in range(5):
            for key in key_list:
                key_name = '{}{}.{}.{}'.format(prefix, i, j, key)
                output_key_name = '{}.{}.{}.{}'.format(output_prefix, j, i * 3 + 1, key)
                value = checkpoint_param["model"].pop(key_name)
                checkpoint_param["model"][output_key_name] = value
torch.save(checkpoint_param, save_model_path)
