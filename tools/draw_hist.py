import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os

font = {'family' : 'Times New Roman',
        'size'   : 20}

plt.rc('font', **font)

# sns.set(rc={'figure.figsize':(4,4)})

checkpoint_path = '/home/liujing/Codes/detections/Adelaidet-Quantization/output/fcos/R_18_1x-Full_SyncBN_dorefa_clip_2bit_test/save.pth'
checkpoint = torch.load(checkpoint_path)['input_maps']
output_path = '/home/liujing/Codes/detections/Adelaidet-Quantization/output/fcos/R_18_1x-Full_SyncBN_dorefa_clip_2bit_test/'

# key_prefix = 'proposal_generator.fcos_head.cls_norm0'
key_prefixes = ['proposal_generator.fcos_head.cls_norm', 'proposal_generator.fcos_head.bbox_norm']
for key_prefix in key_prefixes:
    for j in range(4):
        key_p = '{}{}'.format(key_prefix, j)
        plt.grid(True)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        for i in range(5):
            key_name = '{}.{}'.format(key_p, i)
            val = checkpoint[key_name].to('cpu').numpy()
            
            sns.distplot(val.reshape(-1), kde=False, bins=100, label='level-{}'.format(i + 2))
        # plt.yscale('log')
        plt.legend(loc='upper right')
        plt.xlabel("magnitudes")
        plt.ylabel("num")
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, '{}.pdf'.format(key_p)))
        plt.close()
