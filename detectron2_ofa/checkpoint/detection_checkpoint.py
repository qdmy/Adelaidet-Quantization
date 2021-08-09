# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import pickle, os
from fvcore.common.checkpoint import Checkpointer
from fvcore.common.file_io import PathManager

import detectron2_ofa.utils.comm as comm

from .c2_model_loading import align_and_update_state_dicts


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        none_list = []
        for k, v in checkpointables.items():
            if v is None:
                none_list.append(k)
        for k in none_list:
            checkpointables.pop(k)
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            if 'state_dict' in loaded:
                loaded = {"model": loaded['state_dict']}
                loaded["matching_heuristics"] = True
                self.logger.info("Loading model from state_dict of the checkpoint, turn on the matching_heuristics option")
            else:
                loaded = {"model": loaded}
        if "lpf" in filename:
            loaded["matching_heuristics"] = True
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        super()._load_model(checkpoint)

    def load_extra(self, filename):
        if not os.path.isfile(filename):
            filename = PathManager.get_local_path(filename)
        if not os.path.isfile(filename):
            return None

        self.logger.info("Loading model from {}".format(filename))
        checkpoint = super()._load_file(filename)  # load native pth checkpoint
        self._load_model(checkpoint)
        return checkpoint

    def has_checkpoint(self, file_path):
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(file_path, "last_checkpoint")
        return PathManager.exists(save_file)

    def get_checkpoint_file(self, file_path):
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(file_path, "last_checkpoint")
        try:
            with PathManager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(file_path, last_saved)

    def resume_or_load2(self, path: str, output_path: str, *, resume: bool = True):
        """
        If `resume` is True, this method attempts to resume from the last
        checkpoint, if exists. Otherwise, load checkpoint from the given path.
        This is useful when restarting an interrupted training job.

        Args:
            path (str): path to the checkpoint.
            resume (bool): if True, resume from the last checkpoint if it exists.

        Returns:
            same as :meth:`load`.
        """
        if resume and self.has_checkpoint(output_path):
            path = self.get_checkpoint_file(output_path)
        return self.load(path)
