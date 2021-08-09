# AdelaiDet

AdelaiDet is an open source toolbox for multiple instance-level detection applications based on [Detectron2](https://github.com/facebookresearch/detectron2).

## Models

Inference speed is measured by `tools/train_net.py --eval-only`, with batch size 1 on a single 1080Ti GPU.

### COCO Object Detecton Baselines with FCOS

Name |inference time (ms/im) | box AP | download
--- |:---:|:---:|:---:
[fcos_R_50_1x](configs/FCOS-COCO-Detection/fcos_R_50_1x.yaml) |  | 38.7 |
[fcos_R_101_2x](configs/FCOS-COCO-Detection/fcos_R_101_2x.yaml) |  | 42.6 |
[fcos_X_101_2x](configs/FCOS-COCO-Detection/fcos_X_101_2x.yaml) |  | 43.7 |

### COCO Instance Segmentation Baselines with BlendMask

Model | Name |inference time (ms/im) | box AP | mask AP | download
--- |:---:|:---:|:---:|:---:|:---:
Mask R-CNN | [550_R_50_3x](configs/COCO-InstanceSegmentation/mask_rcnn_550_R_50_FPN_3x.yaml) | 63 | 39.1 | 35.3 |
BlendMask | 550_R_50_3x | 40 | 38.8 | 34.3 | [model](https://cloudstor.aarnet.edu.au/plus/s/gW2fgtimaHbRydk/download)
Mask R-CNN | [R_50_1x](configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml) | 90 | 38.6 | 35.2 |
BlendMask | [R_50_1x](configs/BlendMask-InstanceSegmentation/R_50_aux_1x.yaml) | 83 | 39.7 | 35.4 | [model](https://cloudstor.aarnet.edu.au/plus/s/FgT2o0bkhhFTbVc/download)
Mask R-CNN | [R_50_3x](configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml) |  | 41.0 | 37.2 | 
BlendMask | [R_50_3x](configs/BlendMask-InstanceSegmentation/R_50_aux_3x.yaml) |  | 42.5 | 37.6 | [model](https://cloudstor.aarnet.edu.au/plus/s/0vL8TWjh0xYx7O5/download)
Mask R-CNN | [R_101_3x](configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml) |  | 42.9 | 38.6 |
BlendMask | [R_101_3x](configs/BlendMask-InstanceSegmentation/R_101_aux_3x.yaml) |  | 44.7 | 39.5 | [model](https://cloudstor.aarnet.edu.au/plus/s/gcLpLlvZoX2PZLv/download)

### COCO Panoptic Segmentation Baselines with BlendMask
Model | Name | PQ | PQ<sup>Th</sup> | PQ<sup>St</sup> | download
--- |:---:|:---:|:---:|:---:|:---:
Panoptic FPN | [R_50_3x](configs/COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml) | 41.5 | 48.3 | 31.2 | 
BlendMask | [R_50_3x](configs/BlendMask-InstanceSegmentation/Panoptic/panoptic_R_50_3x.yaml) | 42.5 | 49.5 | 32.0 | [model](https://cloudstor.aarnet.edu.au/plus/s/bG0IhYeMAvlTGTq/download)
Panoptic FPN | [R_101_3x](configs/COCO-InstanceSegmentation/panoptic_fpn_R_101_3x.yaml) | 43.0 | 49.7 | 32.9 |
BlendMask | [R_101_3x](configs/BlendMask-InstanceSegmentation/Panoptic/panoptic_R_101_3x.yaml) | 44.3 | 51.6 | 33.2 | [model](https://cloudstor.aarnet.edu.au/plus/s/AEwbhyQ9F3lqvsz/download)

## Installatioin

See [INSTALL.md](INSTALL.md).

## Quick Start

### Inference with Pre-trained Models

1. Pick a model and its config file, for example, `R_50_aux_3x.yaml`.
2. Download the model `wget https://cloudstor.aarnet.edu.au/plus/s/gcLpLlvZoX2PZLv/download -O weights/blendmask_R_101_3x.pth`
3. Run the demo with
```
python demo/demo.py --config-file configs/BlendMask-InstanceSegmentation/R_50_aux_3x.yaml \
  --input input1.jpg input2.jpg \
	[--other-options]
  --opts MODEL.WEIGHTS weights/blendmask_R_101_3x.pth
```

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md#AdelaiDet](datasets/README.md#expected-dataset-structure-for-adelaidet-instance-detection),
then run:
```
python tools/train_net.py --num-gpus 4 \
	--config-file configs/BlendMask-InstanceSegmentation/R_50_aux_1x.yaml
```

The configs are made for 4-GPU training. To train on 1 GPU, change the batch size with:
```
python tools/train_net.py \
	--config-file configs/BlendMask-InstanceSegmentation/R_50_aux_1x.yaml \
	SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.0025
```

## Citing AdelaiDet

If you use this toolbox in your research or wish to refer to the baseline results, please use the following BibTeX entries.

```BibTeX
@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}

@inproceedings{Chen2020Blendmask,
  title   =  {Blendmask: Top-Down Meets Bottom-Up for Instance Segmentation},
  author  =  {Hao Chen and Kunyang Sun and   Zhi Tian and   Chunhua Shen and   Yongming Huang and Youliang Yan},
  booktitle =  {Working paper},
  year    =  {2020}
}
```


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors.
