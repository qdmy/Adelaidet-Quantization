import torch

def load_bn(path, model):
    checkpoint_param = torch.load(path)
    checkpoint = checkpoint_param['model']
    prefix = "proposal_generator.fcos_head"
    key_name_list = ["weight", "bias", "running_mean", "running_var"]
    head_list = ["cls_norm", "bbox_norm"]
    for i in range(4):
        for j in range(5):
            for head in head_list:
                for weight_name in key_name_list:
                    key = "{}.{}{}.{}.{}".format(prefix, head, i, j, weight_name)
                    value = checkpoint[key]
                    weight = getattr(getattr(model.proposal_generator.fcos_head, "{}{}".format(head, i))[j], weight_name)
                    weight.data = value.data.clone()
    return model