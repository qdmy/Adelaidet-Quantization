# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import random
import torch
import torch.utils.data as data
import numpy as np
from detectron2_ofa.utils.serialize import PicklableWrapper
from detectron2_ofa.data.datasets.coco import get_class_to_superclass
__all__ = ["MapDataset", "DatasetFromList"]


class MapDataset(data.Dataset):
    """
    Map a function over the elements in a dataset.

    Args:
        dataset: a dataset where map function is applied.
        map_func: a callable which maps the element in dataset. map_func is
            responsible for error handling, when error happens, it needs to
            return None so the MapDataset will randomly use other
            elements from the dataset.
    """

    def __init__(self, dataset, map_func):
        self._dataset = dataset
        self._map_func = PicklableWrapper(map_func)  # wrap so that a lambda will work

        self._rng = random.Random(42)
        self._fallback_candidates = set(range(len(dataset)))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        retry_count = 0
        cur_idx = int(idx)

        while True:
            dataset_dict = self._dataset[cur_idx][0]
            # super_targets_mask = self._dataset[cur_idx][1]
            # super_targets_inverse_mask = self._dataset[cur_idx][2]
            # super_targets_idx = self._dataset[cur_idx][3]
            # super_target = self._dataset[cur_idx][4]

            data = self._map_func(dataset_dict)
            if data is not None:
                self._fallback_candidates.add(cur_idx)
                return (data, *self._dataset[cur_idx][1:]) # 后面四项数据没包装进去，dataloader获取不到

            # _map_func fails for this idx, use a random new index from the pool
            retry_count += 1
            self._fallback_candidates.discard(cur_idx)
            cur_idx = self._rng.sample(self._fallback_candidates, k=1)[0]

            if retry_count >= 3:
                logger = logging.getLogger(__name__)
                logger.warning(
                    "Failed to apply `_map_func` for idx: {}, retry count: {}".format(
                        idx, retry_count
                    )
                )


class DatasetFromList(data.Dataset):
    """
    Wrap a list to a torch Dataset. It produces elements of the list as data.
    """

    def __init__(self, lst: list, ranges, meta, in_hier=None, copy: bool = True):
        """
        Args:
            lst (list): a list which contains elements to produce.
            copy (bool): whether to deepcopy the element when producing it,
                so that the result can be modified in place without affecting the
                source in the list.
        """
        self._lst = lst
        self._copy = copy
        self.meta = meta
        self.in_hier = in_hier
        # 下面的数据都是针对每一个物体的，按照图片的顺序，然后每张图片按照其中物体的顺序，与分类任务还是不一样的
        # create a dict contain the start and end anno of each image's objects
        self.targets = []
        self.images_part = []  # 保存每张图片对应的obj是list中的哪一段，原本是定义为dict，但是因为torch的dataset处理getitem的时候是在触发到indexerror时自动停止，结果我这里先遇到了一个keyerror，所以它就报错，改为list，就会在idx超出范围时报错indexerror，就会停止读取数据集了
        end = 0  # 每张图片中包含的obj在list中的end index，最后的数值大小是所有物体的数量
        for i, s in enumerate(self._lst):
            start = end  # after a image is done, the end of the last one becomes the start of next one
            for j, obj in enumerate(s['annotations']):
                self.targets.append(obj["category_id"])
                end += 1
            self.images_part.append([start, end])

        # 这里获取的targets已经是0-80的label了，不需要映射了
        # self.category_ids = self.targets

        self.class_to_superclass = np.ones(len(self.in_hier.in_wnids_child)) * -1 # 应该是一个长度为80的array

        for ran in ranges: # ranges里保存的是连续的id，是属于0-80范围的
            for classnum in ran:
                classname = self.in_hier.num_to_name_child[classnum]
                classwnid = self.in_hier.name_to_wnid_child[classname]
                parentwnid = self.in_hier.tree_child[classwnid].parent_wnid
                parentnum = self.in_hier.wnid_to_num_parent[parentwnid]
                self.class_to_superclass[classnum] = parentnum

        # 验证一下一致性，之前有定义一个连续id到超类id的字典
        for num, super_idx in self.meta.class_to_superclass_idx.items():
            assert self.class_to_superclass[num] == super_idx, 'inconsistency between num and superclass idx projection'

        # self.super_targets里不应该有-1
        self.super_targets = self.class_to_superclass[self.targets]

        self.n_superclass = len(ranges)
        self.super_targets_masks = (self.super_targets.reshape(-1, 1) == self.class_to_superclass).astype("single")
        self.super_targets_inverse_masks = (self.super_targets.reshape(-1, 1) != self.class_to_superclass).astype(
            "single")
        self.super_targets_idxes = []
        for idx in range(end):
            self.super_targets_idxes.append((self.super_targets[idx] == self.class_to_superclass).nonzero()[0])
        self.super_targets_idxes = np.stack(self.super_targets_idxes, axis=0).astype(
            "int64")  # 不知道是不是所有类的图片数量都一样，这里可能有问题

        self.superclass_masks = []
        self.superclass_samples_indices = []
        for i in range(self.n_superclass):
            idx = (self.super_targets == i).nonzero()[0]
            self.superclass_samples_indices.append(idx.tolist())
            superclass_mask = (self.class_to_superclass == i).astype("single")
            self.superclass_masks.append(superclass_mask)

    def __len__(self):
        return len(self._lst)

    def __getitem__(self, idx):
        # if idx >= len(self):
        #     raise IndexError()
        start, end = self.images_part[idx]
        if self._copy:
            return copy.deepcopy(self._lst[idx]), copy.deepcopy(self.super_targets_masks[start:end]), copy.deepcopy(self.super_targets_inverse_masks[start:end]),\
                copy.deepcopy(self.super_targets_idxes[start:end]), copy.deepcopy(self.super_targets[start:end])
        else:
            return self._lst[idx], self.super_targets_masks[start:end], self.super_targets_inverse_masks[start:end],\
                self.super_targets_idxes[start:end], self.super_targets[start:end]
