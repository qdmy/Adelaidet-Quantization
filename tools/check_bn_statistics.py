import torch

pretrained_model_path = "/home/liujing/Models/chenpeng/R_18_1x-Full_BN/model_final.pth"

# load pre-trained model into pytorch_model
checkpoint_param = torch.load(pretrained_model_path, map_location='cpu')
model = checkpoint_param['model']
# prefiex = 'proposal_generator.fcos_head.cls_norm'
prefiex = 'proposal_generator.fcos_head.bbox_norm'

for j in range(4):
    for i in range(5):
        weight = model['{}{}.{}.weight'.format(prefiex, j, i)].mean()
        bias = model['{}{}.{}.bias'.format(prefiex, j, i)].mean()
        mean = model['{}{}.{}.running_mean'.format(prefiex, j, i)].mean()
        var = model['{}{}.{}.running_var'.format(prefiex, j, i)].mean()
        print('j:{}, i:{}, weight: {:.4f}, bias: {:.4f}, mean: {:.4f}, var: {:.4f}'.format(j, i, weight, bias, mean, var))