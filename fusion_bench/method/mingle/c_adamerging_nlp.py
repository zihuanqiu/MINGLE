import logging
import random
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

import lightning as L
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel, get_layer_wise_weights, merge_and_unload)
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.json import save_to_json
from fusion_bench.utils.type import TorchModelType

log = logging.getLogger(__name__)


def del_attr(obj, names: List[str]):
    """
    Deletes an attribute from an object recursively.

    Args:
        obj (object): Object to delete attribute from.
        names (list): List of attribute names to delete recursively.
    """
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names: List[str], val):
    """
    Sets an attribute of an object recursively.

    Args:
        obj (object): Object to set attribute of.
        names (list): List of attribute names to set recursively.
        val (object): Value to set the attribute to.
    """
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)


def get_attr(obj, names: List[str]):
    """
    Gets an attribute of an object recursively.

    Args:
        obj (object): Object to get attribute of.
        names (list): List of attribute names to get recursively.

    Returns:
        object: The attribute of the object.
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def entropy_loss(logits: Tensor) -> Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()


def _select_seed_subset(
    dataset,
    k: int | None = 100,  # 新增：想要抽取的条数
    rng: random.Random | None = None,  # 新增：可选的随机数发生器（用于可复现）
):
    """
    简化版：无视标签，随机抽取 k 条样本（默认 100）。当 k>=len(dataset) 时返回全量。
    """
    n = len(dataset)
    if k is None or k <= 0 or k >= n:
        return dataset
    r = rng if rng is not None else random
    idxs = r.sample(range(n), k)
    return Subset(dataset, idxs)


def append_finetuned(
    merged: LayerWiseMergedModel, new_model: nn.Module, init_value: float
) -> LayerWiseMergedModel:
    """
    Append a new finetuned model to the merged wrapper,
    expanding the layer_wise_weight accordingly,
    and convert new_model ==> delta = new_model - pretrained_model.
    """
    # 1) deepcopy + freeze
    new_model = deepcopy(new_model).requires_grad_(False)

    # 2) compute delta: for every param in pretrained_model
    base = merged.pretrained_model
    for name, p_base in base.named_parameters():
        parts = name.split(".")
        # subtract: delta_param = p_finetuned - p_base
        p_delta = get_attr(new_model, parts)
        p_delta.data = p_delta.data - p_base.data

    # 3) append to module list
    merged.task_vectors.append(new_model)

    # 4) expand merge_weight from (k x N) to ((k+1) x N)
    old_w = merged.merge_weight.data  # shape [k, num_layers]
    new_row = torch.full(
        (1, old_w.size(1)), init_value, dtype=old_w.dtype, device=old_w.device
    )
    new_w = torch.cat([old_w, new_row], dim=0)
    merged.merge_weight = nn.Parameter(new_w, requires_grad=True)

    return merged


class ContinualLayerWiseAdaMerging(BaseAlgorithm, LightningFabricMixin):
    """
    Continual (sequential) layer-wise AdaMerging algorithm.

    At each step, appends one fine-tuned model,
    updates merging weights via optional TTA,
    and proceeds without revisiting prior models.
    """

    def __init__(
        self,
        init_values: float = None,
        weights: Optional[str] = None,
        clamp_weights: bool = True,
        tie_weights: bool = False,
        strict: bool = True,
        normalized_merging_weights: bool = False,
        lr: float = 1e-3,
        batch_size: int = 16,
        max_steps: int = 50,
        shuffle_order: bool = True,
        fast_dev_run: bool = True,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        seed_buffer_size: int = 100,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.init_values = init_values
        self.weights = weights
        self.clamp_weights = clamp_weights
        self.tie_weights = tie_weights
        self.strict = strict
        self.normalized_merging_weights = normalized_merging_weights
        self.lr = lr
        self.max_steps = max_steps
        self.shuffle_order = shuffle_order
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self.seed_buffer_size = seed_buffer_size
        self.seed = seed
        self.batch_size = batch_size
        self.fast_dev_run = fast_dev_run
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        names = modelpool.model_names.copy()
        if self.shuffle_order:
            random.shuffle(names)

        accelerator = self.fabric.device
        self.taskpool = self._program.taskpool  # generic, not casting

        # Load pretrained once
        pretrained = modelpool.load_model("_pretrained_").to(accelerator)

        num_layers = len(
            tuple(filter(lambda p: p.requires_grad, pretrained.parameters()))
        )

        # Sequentially append and adapt
        for step, task in enumerate(names):

            init_val = self.init_values or 1.0
            init_w = get_layer_wise_weights(
                num_models=1, num_layers=num_layers, init_values=init_val
            )

            ft_model = modelpool.load_model(task).to(accelerator)

            merged = LayerWiseMergedModel(
                layer_wise_weight=init_w,
                pretrained_model=deepcopy(pretrained),
                finetuned_models=[ft_model],
                clamp_weights=self.clamp_weights,
                tie_weights=self.tie_weights,
                strict=self.strict,
                normalized_merging_weights=self.normalized_merging_weights,
            )

            merged = self._test_time_adapt_nlp(
                merged, task.replace("glue-", ""), lr=self.lr
            )

            merged = merged.merge_and_unload()

            if self.save_on_every_step:
                self.save_merged_model(merged, step)

            if self.evaluate_on_every_step or step == len(names) - 1:
                self.taskpool._is_setup = False
                seen_task_names = [mn.replace("glue-", "") for mn in names[: step + 1]]
                self.taskpool._all_task_names = seen_task_names
                report = self.taskpool.evaluate(deepcopy(merged))
                save_to_json(report, Path(self.log_dir) / f"report_{step}.json")

        return merged

    def compute_logits(
        self,
        module,
        batch,
    ) -> Tensor:
        """
        Compute the logits for the given images and task.

        Args:
            module: The model module.
            images (Tensor): The input images.
        Returns:
            Tensor: The computed logits.
        """
        input_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]

        # remove padding tokens from the input
        while attention_mask[:, -1].eq(0).all():
            input_ids = input_ids[:, :-1].to(self.fabric.device)
            attention_mask = attention_mask[:, :-1].to(self.fabric.device)

        outputs = module(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=torch.ones(
                input_ids.size(0), 1, dtype=torch.long, device=input_ids.device
            ),
        )
        logits = outputs.logits[:, 0, :]
        return logits

    def _test_time_adapt_nlp(
        self, merged: T5ForConditionalGeneration, task_name: str, lr: float
    ):
        task = self.taskpool.load_task(task_name)
        dataset = task.test_dataset

        buf_k = int(self.seed_buffer_size)
        rng = (
            random.Random(self.seed)
            if getattr(self, "seed", None) is not None
            else None
        )

        dataset = _select_seed_subset(dataset, k=buf_k, rng=rng)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=(
                getattr(task, "test_loader").collate_fn
                if hasattr(task, "test_loader")
                else None
            ),
        )

        merged = merged.to(self.fabric.device)
        merged.train()

        named_trainables = [
            (n, p) for n, p in merged.named_parameters() if p.requires_grad
        ]
        optim = torch.optim.Adam([p for _, p in named_trainables], lr=lr)
        id2name = {id(p): n for n, p in named_trainables}

        steps = self.max_steps
        it = iter(loader)
        pbar = tqdm(range(steps), desc=f"TTA(NLP) @ {task_name}", dynamic_ncols=True)
        merged.merge_weights()
        for _ in pbar:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            logits = self.compute_logits(merged, batch)
            logits = logits.mean(dim=0, keepdim=True)
            loss = entropy_loss(logits)
            loss.backward()

            print("========= check grad =========")
            for n, p in merged.named_parameters():
                if p.requires_grad:
                    print(f"Name: {n}, Shape: {p.shape}, Grad: {p.grad}")
            print("========================================")

            optim.step()
            optim.zero_grad()
            merged.merge_weights()
            pbar.set_postfix({"loss": loss.item()})

        torch.cuda.empty_cache()
        return merged
