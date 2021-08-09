import torch
import os
from matplotlib import pyplot as plt
from detectron2.modeling.quantization.dorefa_clip import quantize_activation

def draw_fig(vis_input, prefix, type_name, name):
    vis_input = vis_input
    np_img = vis_input.cpu().numpy()
    np_img = np_img[0]
    n,h,w = np_img.shape

    # write to file
    for i in range(n):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(3, 3)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(np_img[i], cmap='jet')
        output_path = '{}/{}/{}/'.format(prefix, type_name, name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        fig.savefig('{}{}.png'.format(output_path, i))

if __name__ == "__main__":
    save_path = '/home/liujing/Codes/detections/Adelaidet-Quantization/output/fcos/R_18_1x-Full_SyncBN_dorefa_clip_2bit_test/save.pth'
    output_path = '/home/liujing/Codes/detections/Adelaidet-Quantization/output/fcos/R_18_1x-Full_SyncBN_dorefa_clip_2bit_test/'
    save_dict = torch.load(save_path, map_location='cpu')
    input_maps = save_dict['input_maps']
    feature_maps = save_dict['feature_maps']
    weight_clip_value = save_dict['weight_clip_value']
    activation_clip_value = save_dict['activation_clip_value']

    # for key, value in feature_maps.items():
        # draw_fig(value, output_path, "feature_maps", key)

    # print(activation_clip_value.keys())

    for key, value in input_maps.items():
        # draw_fig(value, output_path, "input_maps", key)
        if key in activation_clip_value:
            print(value)
            quantized_input = quantize_activation(value, 2, activation_clip_value[key]).detach()
            print(quantized_input)
            assert False
            # draw_fig(quantized_input, output_path, "qinput_maps", key)