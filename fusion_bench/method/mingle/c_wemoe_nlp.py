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
from transformers import T5ForConditionalGeneration
from transformers.activations import GELUActivation
from transformers.models.clip.modeling_clip import CLIPEncoder
from transformers.models.t5.modeling_t5 import (T5Attention,
                                                T5DenseGatedActDense)

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import CLIPClassificationMixin, LightningFabricMixin
from fusion_bench.models.hf_clip import HFCLIPClassifier
from fusion_bench.models.we_moe import (WeightEnsemblingMoE,
                                        construct_weight_ensembling_gate)
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.utils.data import InfiniteDataLoader
from fusion_bench.utils.json import save_to_json
from fusion_bench.utils.state_dict_arithmetic import (state_dict_add,
                                                      state_dict_mul,
                                                      state_dict_sub)

from .utils import (frobenius_inner_product, get_task_vector_norm,
                    is_leaf_module, svd)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


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


class LoRAT5DenseGatedActDense(nn.Module):
    """
    LoRA adapter for T5DenseGatedActDense.
    Collects the weight deltas for wi_0, wi_1 and wo projections and
    exposes them via get_lora_deltas(), so your merging routine can
    pick them up by name and fuse automatically.
    """

    def __init__(self, base_ff, expert_ff, rank):
        super().__init__()
        # LoRA on the first gate projection (wi_0)
        in_dim0, hid_dim0 = (
            base_ff.wi_0.in_features,
            base_ff.wi_0.out_features,
        )
        self.lora_wi_0 = LoRA(
            torch.zeros((rank, in_dim0)),
            torch.zeros((hid_dim0, rank)),
        )
        delta_w0 = expert_ff.wi_0.weight.data - base_ff.wi_0.weight.data
        self.lora_wi_0.set_delta(delta_w0, rank)

        # LoRA on the second gate projection (wi_1)
        in_dim1, hid_dim1 = (
            base_ff.wi_1.in_features,
            base_ff.wi_1.out_features,
        )
        self.lora_wi_1 = LoRA(
            torch.zeros((rank, in_dim1)),
            torch.zeros((hid_dim1, rank)),
        )
        delta_w1 = expert_ff.wi_1.weight.data - base_ff.wi_1.weight.data
        self.lora_wi_1.set_delta(delta_w1, rank)

        # LoRA on the output projection (wo)
        in_dim2, out_dim2 = (
            base_ff.wo.in_features,
            base_ff.wo.out_features,
        )
        self.lora_wo = LoRA(
            torch.zeros((rank, in_dim2)),
            torch.zeros((out_dim2, rank)),
        )
        delta_wo = expert_ff.wo.weight.data - base_ff.wo.weight.data
        self.lora_wo.set_delta(delta_wo, rank)

    def get_lora_deltas(self) -> dict:
        """
        Returns a dict mapping the exact keys in a
        T5DenseGatedActDense's state_dict to the LoRA deltas.
        """
        return {
            "wi_0.weight": self.lora_wi_0.get_delta().detach(),
            "wi_1.weight": self.lora_wi_1.get_delta().detach(),
            "wo.weight": self.lora_wo.get_delta().detach(),
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
    old_E = old_fc.out_features  # 之前的专家数
    num_E = old_E + 1  # 新的专家数

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
    """
    Merge all T5 attention layers in `merged_model` by adding
    scaling_factor * (finetuned – pretrained) to each attention
    parameter in the merged model.
    """
    # Iterate over every submodule in the merged model
    for name, m_attn in merged_model.named_modules():
        # Select only T5Attention layers (covers both self- and cross-attn)
        if isinstance(m_attn, T5Attention):
            # Fetch the corresponding attention modules by name
            p_attn = pretrained_model.get_submodule(name)
            f_attn = finetuned_model.get_submodule(name)

            # Build dicts for quick param lookup
            p_params = dict(p_attn.named_parameters())
            f_params = dict(f_attn.named_parameters())

            # For each parameter in the merged attention, apply arithmetic merge
            for param_name, m_param in m_attn.named_parameters():
                p_param = p_params[param_name]
                f_param = f_params[param_name]
                delta = f_param.data - p_param.data
                m_param.data = m_param.data + scaling_factor * delta

    return merged_model


class ContinualWEMoE(BaseAlgorithm, LightningFabricMixin):
    def __init__(
        self,
        init_lambda: float,
        router_hidden_layers: int = 2,
        seed_buffer_size: int = 5,
        max_steps: int = 50,
        batch_reduce: str = "mean",
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        use_tta: bool = False,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        lr: float = 1e-3,
        batch_size: int = 16,
        **kwargs,
    ):
        self.init_lambda = init_lambda
        self.router_hidden_layers = router_hidden_layers
        self.batch_reduce = batch_reduce
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.seed_buffer_size = seed_buffer_size
        self.use_tta = use_tta
        self.max_steps = max_steps
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self.lr = lr
        self.batch_size = batch_size

        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        names = modelpool.model_names.copy()
        if self.shuffle_order:
            random.shuffle(names)

        accelerator = self.fabric.device
        self.taskpool = self._program.taskpool  # generic, not casting

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

            for name, base_ff in base.named_modules():
                # detect every T5 MLP layer
                if isinstance(base_ff, T5DenseGatedActDense):
                    # grab the corresponding fine-tuned MLP
                    expert_ff = tm.get_submodule(name)

                    lora_expert = LoRAT5DenseGatedActDense(
                        base_ff, expert_ff, rank=32
                    ).to(accelerator)

                    # locate the same module in the MoE model
                    moe_ff = moe_model.get_submodule(name)

                    # build or extend the MoE for this FF layer
                    if step == 0:
                        new_moe_ff = SkipTVWEMoE(
                            hidden_size=base.config.d_model,
                            base_model=base_ff,
                            expert_models=[lora_expert],
                            init_lambda=self.init_lambda,
                            batch_first=True,
                            router_hidden_layers=self.router_hidden_layers,
                            batch_reduce=self.batch_reduce,
                        )
                    else:
                        new_moe_ff = append_expert_and_rebuild_gate(
                            moe_ff,
                            lora_expert,
                            self.init_lambda,
                            accelerator,
                        )

                    # replace the module in moe_model:
                    parent_name, attr = name.rsplit(".", 1)
                    parent = (
                        moe_model.get_submodule(parent_name)
                        if parent_name
                        else moe_model
                    )
                    parent._modules[attr] = new_moe_ff

            if self.use_tta:
                moe_model = self._test_time_adapt_nlp(
                    moe_model, task.replace("glue-", ""), lr=self.lr
                )

            if self.save_on_every_step:
                self.save_merged_model(moe_model, step)

            if self.evaluate_on_every_step or step == len(names) - 1:
                self.taskpool._is_setup = False
                seen_task_names = [mn.replace("glue-", "") for mn in names[: step + 1]]
                self.taskpool._all_task_names = seen_task_names
                report = self.taskpool.evaluate(deepcopy(moe_model))
                save_to_json(report, Path(self.log_dir) / f"report_{step}.json")

        return moe_model

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

            optim.step()
            optim.zero_grad()
            pbar.set_postfix({"loss": loss.item()})

        torch.cuda.empty_cache()
        return merged
