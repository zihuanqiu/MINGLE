import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import CLIPVisionModel

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.utils.state_dict_arithmetic import (state_dict_add,
                                                      state_dict_mul,
                                                      state_dict_sub)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class ContinualTaskArithmeticForT5(BaseAlgorithm, LightningFabricMixin):
    def __init__(
        self,
        scaling_factor: float,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        **kwargs,
    ):
        """
        Continual Model Merging via Task Arithmetic.

        Args:
            scaling_factor (float): the scaling factor to use.
            shuffle_order (bool): whether to shuffle the order of the models.
            seed (Optional[int]): the seed to use.
            save_on_every_step (bool): whether to save the merged model on every step.
            evaluate_on_every_step (bool): whether to evaluate the merged model on every step.
        """
        self.scaling_factor = scaling_factor
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        super().__init__(**kwargs)

    @torch.no_grad()
    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        accelerator = self.fabric.device
        self.taskpool = self._program.taskpool  # generic, not casting

        # log the model names
        if self.log_dir is not None:
            save_to_json(model_names, Path(self.log_dir) / "model_names.json")
            tensorboard_summarywriter: "SummaryWriter" = self.tensorboard_summarywriter
            tensorboard_summarywriter.add_text(
                "global/model_names", str(model_names), global_step=0
            )

        # get the average model
        pretrained_model = modelpool.load_pretrained_model()
        merged_model = deepcopy(pretrained_model)

        for model_idx, model_name in tqdm(
            enumerate(model_names), desc="Processing models"
        ):
            task_model = modelpool.load_model(model_name)

            task_vector = state_dict_sub(
                task_model.state_dict(keep_vars=True),
                pretrained_model.state_dict(keep_vars=True),
            )

            task_vector = state_dict_mul(task_vector, self.scaling_factor)

            state_dict = state_dict_add(
                pretrained_model.state_dict(keep_vars=True), task_vector
            )

            merged_model.load_state_dict(state_dict)

            if self.save_on_every_step:
                self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step or model_idx == len(model_names) - 1:
                self.taskpool._is_setup = False
                seen_task_names = [
                    mn.replace("glue-", "") for mn in model_names[: model_idx + 1]
                ]
                self.taskpool._all_task_names = seen_task_names
                report = self.taskpool.evaluate(deepcopy(merged_model.to(accelerator)))
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

        return merged_model

    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        torch.save(
            merged_model.state_dict(),
            Path(self.log_dir) / "checkpoints" / f"model_{step}.pth",
        )
