import functools
import logging
import os
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple, cast

import lightning as L
import torch
from lightning.pytorch.profilers import SimpleProfiler
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers.models.clip.modeling_clip import CLIPEncoder

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import CLIPClassificationMixin, LightningFabricMixin
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.we_moe import (WeightEnsemblingMoE,
                                        construct_weight_ensembling_gate)
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils import gpu_mem_context, timeit_context
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.json import save_to_json
from fusion_bench.utils.state_dict_arithmetic import (state_dict_add,
                                                      state_dict_mul,
                                                      state_dict_sub)

from .utils import (frobenius_inner_product, get_task_vector_norm,
                    is_leaf_module, svd)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class LoRA(nn.Module):
    def __init__(self, A, B):
        super().__init__()
        self.A = nn.Parameter(A, requires_grad=False)
        self.B = nn.Parameter(B, requires_grad=False)

    def forward(self, x):
        x = nn.functional.linear(x, self.A)
        x = nn.functional.linear(x, self.B)
        return x

    def get_delta(self):
        return self.B @ self.A

    def set_delta(self, delta, rank=None):
        u, s, v = svd(delta)
        if rank is None:
            rank = self.A.data.shape[0]
        self.A = nn.Parameter(torch.diag(s[:rank]) @ v[:, :rank].T, requires_grad=False)
        self.B = nn.Parameter(u[:, :rank], requires_grad=False)


class SkipTVWEMoE(WeightEnsemblingMoE):
    """
    A subclass of WeightEnsemblingMoE that skips the initial "compute the task vectors" phase.
    """

    def __init__(
        self,
        hidden_size: int,
        base_model: nn.Module,
        expert_models: List[nn.Module],
        init_lambda: float = 0.2,
        batch_first: bool = False,
        router_hidden_layers: int = 2,
        batch_reduce: bool = False,
    ):
        # Initialize base attributes and gate
        super(WeightEnsemblingMoE, self).__init__()  # bypass parent's __init__ body
        self.num_experts = len(expert_models)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.batch_reduce = batch_reduce

        self.gate = construct_weight_ensembling_gate(
            hidden_size,
            self.num_experts,
            init_lambda=init_lambda,
            num_hidden_layers=router_hidden_layers,
        )

        # Skip the compute task vectors block.
        # Directly fix base_model and expert_models,
        # and register them as task_vectors.
        self.base_model = base_model.requires_grad_(False)
        for m in expert_models:
            m.requires_grad_(False)
        self.task_vectors = nn.ModuleList(expert_models)

    def merge_weights(self, expert_weights):
        # grab the MLP’s own params
        state_dict = self.base_model.state_dict(keep_vars=True)

        # expert_weights & task_vectors are aligned with your LoRAMLP experts
        for weight, lora_mlp in zip(expert_weights, self.task_vectors):
            for param_key, delta in lora_mlp.get_lora_deltas().items():
                # direct in-place addition on that MLP param
                state_dict[param_key] = state_dict[param_key] + weight * delta

        self._merged_state_dict = state_dict
        return state_dict


class LoRAMLP(nn.Module):
    def __init__(self, base_mlp: nn.Module, expert_mlp: nn.Module, rank: int = 16):
        super().__init__()
        # build LoRA for fc1
        self.lora1 = LoRA(
            torch.zeros((rank, base_mlp.fc1.in_features)),
            torch.zeros((base_mlp.fc1.out_features, rank)),
        )
        delta1 = expert_mlp.fc1.weight.data - base_mlp.fc1.weight.data
        self.lora1.set_delta(delta1, rank)
        # build LoRA for fc2
        self.lora2 = LoRA(
            torch.zeros((rank, base_mlp.fc2.in_features)),
            torch.zeros((base_mlp.fc2.out_features, rank)),
        )
        delta2 = expert_mlp.fc2.weight.data - base_mlp.fc2.weight.data
        self.lora2.set_delta(delta2, rank)

    def get_lora_deltas(self) -> dict:
        """
        Return exactly the keys as they appear in base_model.state_dict(),
        i.e. 'fc1.weight' and 'fc2.weight'.
        """
        return {
            "fc1.weight": self.lora1.get_delta().detach(),
            "fc2.weight": self.lora2.get_delta().detach(),
        }


def entropy_loss(logits: Tensor) -> Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()


def append_expert_and_rebuild_gate(
    merged_mlp: nn.Module, expert_mlp: nn.Module, init_lambda: float, accelerator
):
    merged_mlp.task_vectors.append(expert_mlp.requires_grad_(False))

    old_fc = merged_mlp.gate.fc
    in_feats = old_fc.in_features
    old_E = old_fc.out_features
    num_E = old_E + 1

    old_W = old_fc.weight.data.clone()  # shape (old_E, in_feats)
    old_b = old_fc.bias.data.clone()

    new_fc = nn.Linear(in_feats, num_E, bias=True).to(accelerator)

    nn.init.normal_(new_fc.weight, std=0.01)
    nn.init.constant_(new_fc.bias, init_lambda)

    new_fc.weight.data[:old_E] = old_W
    new_fc.bias.data[:old_E] = old_b

    merged_mlp.gate.fc = new_fc

    return merged_mlp


@torch.no_grad()
def task_arithmetic_merge_attn(
    merged_model: nn.Module,
    pretrained_model: nn.Module,
    finetuned_model: nn.Module,
    scaling_factor: float,
) -> nn.Module:
    encoder_layers = merged_model.vision_model.encoder.layers
    pre_layers = pretrained_model.vision_model.encoder.layers
    fine_layers = finetuned_model.vision_model.encoder.layers

    for idx, (m_layer, p_layer, f_layer) in enumerate(
        zip(encoder_layers, pre_layers, fine_layers)
    ):
        m_attn = m_layer.self_attn
        p_attn = p_layer.self_attn
        f_attn = f_layer.self_attn

        for name, m_param in m_attn.named_parameters():
            p_param = dict(p_attn.named_parameters())[name]
            f_param = dict(f_attn.named_parameters())[name]

            delta = f_param.data - p_param.data
            m_param.data = m_param.data + scaling_factor * delta

    return merged_model


class ContinualWEMoE(BaseAlgorithm, LightningFabricMixin):
    def __init__(
        self,
        init_lambda: float,
        router_hidden_layers: int = 2,
        seed_sample_number: int = 5,
        max_steps: int = 50,
        batch_reduce: str = "mean",
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        use_tta: bool = False,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        **kwargs,
    ):
        self.init_lambda = init_lambda
        self.router_hidden_layers = router_hidden_layers
        self.batch_reduce = batch_reduce
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.seed_sample_number = seed_sample_number
        self.use_tta = use_tta
        self.max_steps = max_steps
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self._config = DictConfig(kwargs)
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

        base = modelpool.load_pretrained_model().to(accelerator)
        moe_model = deepcopy(base)
        moe_model.requires_grad_(False)

        if self.log_dir is not None:
            save_to_json(names, Path(self.log_dir) / "model_names.json")
            writer: "SummaryWriter" = self.tensorboard_summarywriter
            writer.add_text("global/model_names", str(names), global_step=0)

        for step, task in enumerate(tqdm(names, desc="WEMoE merging")):
            tm = modelpool.load_model(task).to(accelerator)

            # Merge the models using task arithmetic
            moe_model = task_arithmetic_merge_attn(
                deepcopy(moe_model),
                deepcopy(base),
                deepcopy(tm),
                scaling_factor=self.init_lambda,
            ).requires_grad_(False)

            # Up-scale MLP modules
            base_encoder: CLIPEncoder = base.vision_model.encoder
            moe_encoder: CLIPEncoder = moe_model.vision_model.encoder

            if step == 0:
                num_layers = len(base_encoder.layers)
                for layer_idx in range(num_layers):
                    base_mlp = base_encoder.layers[layer_idx].mlp
                    expert_mlp = tm.vision_model.encoder.layers[layer_idx].mlp

                    lora_expert = LoRAMLP(base_mlp, expert_mlp, rank=64).to(accelerator)

                    # lora-wemoe
                    moe_encoder.layers[layer_idx].mlp = SkipTVWEMoE(
                        hidden_size=base.config.hidden_size,
                        base_model=base_mlp,
                        expert_models=[lora_expert],
                        init_lambda=self.init_lambda,
                        batch_first=True,  # For open_clip models this is False
                        router_hidden_layers=self.router_hidden_layers,
                        batch_reduce=self.batch_reduce,
                    )

                    # wemoe
                    # moe_encoder.layers[layer_idx].mlp = WeightEnsemblingMoE(
                    #     hidden_size=base.config.hidden_size,
                    #     base_model=base_mlp,
                    #     expert_models=[expert_mlp],
                    #     init_lambda=self.init_lambda,
                    #     batch_first=True,  # For open_clip models this is False
                    #     router_hidden_layers=self.router_hidden_layers,
                    #     batch_reduce=self.batch_reduce,
                    # )

            else:
                num_layers = len(base_encoder.layers)
                for layer_idx in range(num_layers):
                    expert_mlp = tm.vision_model.encoder.layers[layer_idx].mlp
                    base_mlp = base_encoder.layers[layer_idx].mlp
                    lora_expert = LoRAMLP(base_mlp, expert_mlp, rank=16).to(accelerator)

                    # lora-wemoe
                    moe_encoder.layers[layer_idx].mlp = append_expert_and_rebuild_gate(
                        moe_encoder.layers[layer_idx].mlp,
                        lora_expert,
                        self.init_lambda,
                        accelerator,
                    )
                    # wemoe
                    # moe_encoder.layers[layer_idx].mlp = append_expert_and_rebuild_gate(moe_encoder.layers[layer_idx].mlp, base_mlp, self.init_lambda, accelerator)

            gate_params = [
                (n, p) for n, p in moe_model.named_parameters() if "gate" in n
            ]

            total_gate_params = sum(p.numel() for n, p in gate_params)
            trainable_gate_params = sum(
                p.numel() for n, p in gate_params if p.requires_grad
            )
            frozen_gate_params = total_gate_params - trainable_gate_params

            print(f"Total gate params: {total_gate_params}")
            print(f"Trainable gate params: {trainable_gate_params}")
            print(f"Frozen gate params: {frozen_gate_params}")

            if self.use_tta:
                moe_model = self.test_time_adaptation(
                    moe_model, tm, task, self.seed_sample_number
                )

            if self.save_on_every_step:
                self.save_merged_model(moe_model, step)

            if self.evaluate_on_every_step or step == len(names) - 1:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {n: self._test_datasets[n] for n in names[: step + 1]}
                )
                report = self.taskpool.evaluate(deepcopy(moe_model))
                save_to_json(report, Path(self.log_dir) / f"report_{step}.json")

        return moe_model

    def test_time_adaptation(
        self, model, task_model, model_name, seed_sample_number=None
    ):
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
            batch_size=self._config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        tta_loader = iter(InfiniteDataLoader(loader))

        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=self._config.lr,
        )

        steps = self.max_steps
        if self._config.fast_dev_run:
            steps = 1
        print("========= Optimized Parameters =========")
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(f"Name: {n}, Shape: {p.shape}, Requires Grad: {p.requires_grad}")
        print("========================================")

        pbar = tqdm(range(steps), "Test-time adaptation", dynamic_ncols=True)

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
                    pbar.set_postfix({"loss": loss.item()})

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
