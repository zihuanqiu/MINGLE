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

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin, SimpleProfilerMixin
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.wrappers.layer_wise_fusion import (
    LayerWiseMergedModel, get_layer_wise_weights, merge_and_unload)
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import gpu_mem_context, timeit_context
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
        seed_sample_number: int = 5,
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
        self.seed_sample_number = seed_sample_number
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
        self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
        self._test_datasets = deepcopy(self.taskpool._test_datasets)

        # Load pretrained once
        pretrained = modelpool.load_model("_pretrained_").to(accelerator)

        emb = pretrained.vision_model.embeddings
        if hasattr(emb, "position_ids"):
            emb.register_buffer("position_ids", emb.position_ids, persistent=True)

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
            emb_ft = ft_model.vision_model.embeddings
            if hasattr(emb_ft, "position_ids"):
                emb_ft.register_buffer(
                    "position_ids", emb_ft.position_ids, persistent=True
                )

            merged = LayerWiseMergedModel(
                layer_wise_weight=init_w,
                pretrained_model=deepcopy(pretrained),
                finetuned_models=[ft_model],
                clamp_weights=self.clamp_weights,
                tie_weights=self.tie_weights,
                strict=self.strict,
                normalized_merging_weights=self.normalized_merging_weights,
            )

            merged = self.test_time_adaptation(merged, task, self.seed_sample_number)
            # merged = self.MTL_test_time_adaptation(merged, names)

            merged = merged.merge_and_unload()

            if self.save_on_every_step:
                self.save_merged_model(merged, step)

            if self.evaluate_on_every_step or step == len(names) - 1:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {n: self._test_datasets[n] for n in names[: step + 1]}
                )
                report = self.taskpool.evaluate(deepcopy(merged))
                save_to_json(report, Path(self.log_dir) / f"report_{step}.json")

        return merged

    def test_time_adaptation(self, model, model_name, seed_sample_number=None):
        self.taskpool._test_datasets = DictConfig(
            {model_name: self._test_datasets[model_name]}
        )
        self.taskpool.setup()

        model = model.to(self.fabric.device)
        self.taskpool.clip_model.vision_model = model
        classifier = HFCLIPClassifier(
            self.taskpool.clip_model, processor=self.taskpool.processor
        )
        classifier = cast(HFCLIPClassifier, self.taskpool.fabric.to_device(classifier))

        classnames, templates = get_classnames_and_templates(model_name)
        classifier.set_classification_task(classnames, templates)

        classifier.train()
        classifier.to(self.fabric.device)
        dataset = self.taskpool.test_datasets[model_name]
        if seed_sample_number is not None:
            dataset = select_seed_subset(dataset, seed_sample_number)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        tta_loader = iter(InfiniteDataLoader(loader))

        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if p.requires_grad], lr=self.lr
        )

        steps = self.max_steps
        if self.fast_dev_run:
            steps = 1
        print("========= Optimized Parameters =========")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"Name: {n}, Shape: {p.shape}, Requires Grad: {p.requires_grad}")
        print("========================================")

        pbar = tqdm(range(steps), "Test-time adaptation", dynamic_ncols=True)

        model.merge_weights()

        with gpu_mem_context("Testing GPU memory usage") as mem_tracker:
            with timeit_context("Merging time"):
                for step_idx in pbar:
                    images, _ = next(tta_loader)
                    images = images.to(self.fabric.device)

                    outputs = classifier(
                        images,
                        return_image_embeds=False,
                        return_dict=True,
                        task_name=model_name,
                    )
                    loss = entropy_loss(outputs["logits"])

                    loss.backward()
                    mem_tracker.update_peak_memory()

                    # print("========= check grad =========")
                    # for n, p in model.named_parameters():
                    #     if p.requires_grad:
                    #         print(f"Name: {n}, Shape: {p.shape}, Grad: {p.grad}")
                    # print("========================================")

                    optimizer.step()
                    optimizer.zero_grad()
                    model.merge_weights()
                    pbar.set_postfix({"loss": loss.item()})

        return model

    def MTL_test_time_adaptation(self, model, model_names):

        self.taskpool._test_datasets = DictConfig(
            {n: self._test_datasets[n] for n in model_names}
        )

        self.taskpool.setup()
        self.taskpool.clip_model.vision_model = model
        classifier = HFCLIPClassifier(
            self.taskpool.clip_model,
            processor=self.taskpool.processor,
        )

        classifiers = {}
        tta_loaders = {}

        for model_name in model_names:
            classifier = cast(
                HFCLIPClassifier, self.taskpool.fabric.to_device(classifier)
            )
            classnames, templates = get_classnames_and_templates(model_name)
            self.taskpool.on_task_evaluation_begin(classifier, model_name)
            classifier.set_classification_task(classnames, templates)
            classifiers[model_name] = deepcopy(classifier)
            classifiers[model_name].clip_model.vision_model = model

            tta_loader = DataLoader(
                dataset=self.taskpool.test_datasets[model_name],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            tta_loader = iter(InfiniteDataLoader(tta_loader))
            tta_loaders[model_name] = tta_loader

        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if p.requires_grad], lr=self.lr
        )

        model.train()

        if self.fast_dev_run:
            log.info("Running fast_dev_run, only one step")
            pbar = tqdm(
                range(1),
                "Test-time adaptation",
                dynamic_ncols=True,
            )
        else:
            pbar = tqdm(
                range(self.max_steps),
                "Test-time adaptation",
                dynamic_ncols=True,
            )
        model.merge_weights()
        for step_idx in pbar:
            for task in model_names:
                images, _ = next(tta_loaders[task])
                images = images.to(self.fabric.device)

                logits = classifiers[task](
                    images,
                    return_image_embeds=True,
                    return_dict=True,
                    task_name=model_name,
                )["logits"]

                loss = entropy_loss(logits)
                loss.backward(retain_graph=True)
                pbar.set_postfix({"loss": loss.item()})

            optimizer.step()
            optimizer.zero_grad()
            model.merge_weights()

        return model


def select_seed_subset(dataset, seed_sample_number):
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        label_to_indices[label].append(idx)

    selected_indices = []
    for label, indices in label_to_indices.items():
        if len(indices) >= seed_sample_number:
            selected_indices.extend(random.sample(indices, seed_sample_number))
        else:
            print(
                f"[Warning] Class {label} has only {len(indices)} samples, using all available samples."
            )
            selected_indices.extend(indices)

    total_samples = len(dataset)
    seed_samples = len(selected_indices)
    ratio = seed_samples / total_samples * 100

    print(
        f"Seed sample number: {seed_samples}, Total test dataset size: {total_samples}"
    )
    print(f"Seed sample ratio: {ratio:.2f}% of the total dataset.")

    return Subset(dataset, selected_indices)
