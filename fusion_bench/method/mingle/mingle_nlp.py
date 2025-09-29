# fusion_bench/method/opcm/mingle_nlp.py
import os
import random
from collections import defaultdict
from collections.abc import Hashable
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional
import math

import lightning as L
import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import T5ForConditionalGeneration

from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.models.lora_moe import LoRAMoE
from fusion_bench.utils.json import save_to_json
from fusion_bench.utils.state_dict_arithmetic import (state_dict_add,
                                                      state_dict_mul,
                                                      state_dict_sub)

from .utils import is_leaf_module, svd


def _select_seed_subset(
    dataset,
    k: int | None = 100,  
    rng: random.Random | None = None,  
):
    n = len(dataset)
    if k is None or k <= 0 or k >= n:
        return dataset
    r = rng if rng is not None else random
    idxs = r.sample(range(n), k)
    return Subset(dataset, idxs)


def _kd_loss(student_logits: Tensor, teacher_logits: Tensor, T: float = 2.0) -> Tensor:
    """
    Token-level soft KD. Shapes: [B, L, V]
    """
    s = torch.log_softmax(student_logits / T, dim=-1)
    t = torch.softmax(teacher_logits / T, dim=-1)
    return torch.mean(torch.sum(-t * s, dim=-1)) * (T * T)


class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, r: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r

        self.A = nn.Parameter(torch.zeros(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor):
        delta = F.linear(x, self.B @ self.A)
        return delta

    def get_delta(self):
        return self.B @ self.A

    def set_delta(self, delta: torch.Tensor, rank: int = None):
        if rank is None:
            rank = self.r
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]
        A_new = torch.diag(S_r) @ Vh_r  # (rank × in_features)
        B_new = U_r  # (out_features × rank)
        self.A = nn.Parameter(A_new, requires_grad=False)
        self.B = nn.Parameter(B_new, requires_grad=False)


# ===================== Algorithm =====================


class MINGLE_NLP(BaseAlgorithm, LightningFabricMixin):
    """
    MINGLE for T5-style seq2seq backbones (e.g., Flan-T5 on GLUE).
    - Build LoRAMoE on linear layers selected by config.lora_layer
    - Test-time adaptation: token-level KD from task-specific teacher to merged student
    """

    def __init__(
        self,
        shuffle_order: bool = True,
        seed: Optional[int] = None,
        save_on_every_step: bool = True,
        evaluate_on_every_step: bool = False,
        **kwargs,
    ):
        self.shuffle_order = shuffle_order
        self.seed = seed
        self.save_on_every_step = save_on_every_step
        self.evaluate_on_every_step = evaluate_on_every_step
        self._config = DictConfig(kwargs)
        super().__init__(**kwargs)

    def run(self, modelpool: BaseModelPool):
        if self.seed is not None:
            L.seed_everything(self.seed)

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        accelerator = self.fabric.device
        self.taskpool = self._program.taskpool  # generic, not casting

        # base model
        pretrained = modelpool.load_pretrained_model().to(accelerator)
        assert isinstance(
            pretrained, T5ForConditionalGeneration
        ), f"Expect a T5ForConditionalGeneration, got {type(pretrained)}"
        merged = deepcopy(pretrained)
        merged.requires_grad_(False)

        # iterate per-task model and merge
        for model_idx, model_name in enumerate(model_names):
            teacher = modelpool.load_model(model_name).to(accelerator)

            for module_name, module in list(teacher.named_modules()):

                if not is_leaf_module(module):
                    continue

                merged_module = merged.get_submodule(module_name)
                previous_merged_tv = None
                if isinstance(merged_module, LoRAMoE):
                    previous_merged_tv = 0
                    for lora in merged_module.task_vectors:
                        previous_merged_tv += lora.get_delta().to(accelerator)
                    merged_module = merged_module.base_model

                do_lora = any(key in module_name for key in self._config.lora_layer)

                if isinstance(module, nn.Linear) and do_lora:
                    loraAB = self.construct_lora(
                        pretrained.get_submodule(module_name).weight,
                        teacher.get_submodule(module_name).weight,
                        rank=self._config.lora_r,
                        accelerator=accelerator,
                        previous_merged_tv=previous_merged_tv,
                    )

                    if model_idx == 0:
                        lora_moe = LoRAMoE(
                            hidden_size=merged_module.weight.data.shape[1],
                            base_model=merged_module,
                            expert_models=[loraAB],
                            batch_first=True,
                            batch_reduce=self._config.batch_reduce,
                        )
                        merged.set_submodule(module_name, lora_moe)
                    else:
                        lora_moe = merged.get_submodule(module_name)
                        lora_moe.add_expert(new_expert_models=[loraAB])

            merged = self._test_time_adapt_nlp(
                merged=merged,
                teacher=teacher,
                task_name=model_name.replace("glue-", ""),
                lr=self._config.get("lr", 1e-3),
            )

            torch.cuda.empty_cache()

            if self.save_on_every_step:
                self._save_ckpt(merged, model_idx)

            if self.evaluate_on_every_step or model_idx == len(model_names) - 1:
                self.taskpool._is_setup = False
                seen_task_names = [
                    mn.replace("glue-", "") for mn in model_names[: model_idx + 1]
                ]
                self.taskpool._all_task_names = seen_task_names
                report = self.taskpool.evaluate(deepcopy(merged.to(accelerator)))
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

        return merged

    # ---------- helpers ----------

    def _save_ckpt(self, model: T5ForConditionalGeneration, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        torch.save(
            model.state_dict(), Path(self.log_dir) / "checkpoints" / f"model_{step}.pth"
        )

    @torch.no_grad()
    def _merge_other_params(
        self,
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        param_name: str,
        accelerator: str = "cpu",
    ):
        dev = merged_W.device
        merged_W = merged_W.to(accelerator)
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)
        task_tv = task_W - pretrained_W
        new_merged = merged_W + self.scaling_factor * task_tv
        return new_merged.to(dev)

    @torch.no_grad()
    def construct_lora(
        self, pretrained_W, task_W, rank, accelerator, previous_merged_tv=None
    ):
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)

        task_tv = task_W - pretrained_W

        if previous_merged_tv is not None:
            u, s, v = svd(previous_merged_tv)
            normed_singular_values = s / torch.sum(s)
            entropy = -torch.sum(
                normed_singular_values * torch.log(normed_singular_values)
            )
            effective_rank = int(torch.exp(entropy))
            projected_task_tv = u.T @ task_tv @ v
            projected_task_tv.diag().fill_(0)

            projected_task_tv[:effective_rank, :effective_rank] = 0
            task_tv = u @ projected_task_tv @ v.T

        lora = LoRA(task_tv.shape[1], task_tv.shape[0], rank)
        lora.set_delta(task_tv)
        return lora

    # ---------- TTA for NLP ----------

    def _test_time_adapt_nlp(
        self,
        merged: T5ForConditionalGeneration,
        teacher: T5ForConditionalGeneration,
        task_name: str,
        lr: float,
    ):
        task = self.taskpool.load_task(task_name)
        dataset = task.test_dataset

        buf_k = int(self._config.get("seed_buffer_size", 100))
        rng = (
            random.Random(self.seed)
            if getattr(self, "seed", None) is not None
            else None
        )

        dataset = _select_seed_subset(dataset, k=buf_k, rng=rng)

        loader = DataLoader(
            dataset,
            batch_size=self._config.batch_size,
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
        teacher = teacher.to(self.fabric.device).eval()
        merged.train()

        named_trainables = [
            (n, p)
            for n, p in merged.named_parameters()
            if p.requires_grad and "gate" in n
        ]
        optim = torch.optim.Adam([p for _, p in named_trainables], lr=lr)
        id2name = {id(p): n for n, p in named_trainables}

        steps = (
            self._config.max_steps if not self._config.get("fast_dev_run", False) else 1
        )
        it = iter(loader)
        pbar = tqdm(range(steps), desc=f"TTA(NLP) @ {task_name}", dynamic_ncols=True)

        for _ in pbar:
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            input_ids = batch["input_ids"].to(self.fabric.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.fabric.device)

            with torch.no_grad():
                gen = teacher.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=min(32, self._config.get("max_new_tokens", 32)),
                    do_sample=False,
                    num_beams=1,
                )
                t_out = teacher(
                    input_ids=input_ids, attention_mask=attention_mask, labels=gen
                )
                t_logits = t_out.logits.detach()

            s_out = merged(
                input_ids=input_ids, attention_mask=attention_mask, labels=gen
            )
            s_logits = s_out.logits

            loss = _kd_loss(s_logits, t_logits, T=self._config.get("kd_T", 2.0))
            loss.backward()

            if self._config.get("constraint_gate", False):
                for module in merged.modules():
                    if isinstance(module, LoRAMoE):
                        for _, p in module.gate.named_parameters():
                            if p.grad is not None:
                                p.grad.data = module.project_gradient(
                                    p.grad.data,
                                    gamma=self._config.gamma,
                                    beta=self._config.beta,
                                    debug=self._config.get("fast_dev_run", False),
                                )

            # for gi, pg in enumerate(optim.param_groups):
            #     for p in pg["params"]:
            #         name = id2name.get(id(p), f"param_group{gi}")
            #         g = p.grad
            #         if g is None:
            #             print(f"[step {step}] {name}: grad=None")
            #         else:
            #             g = g.detach()
            #             try:
            #                 gnorm = g.norm().item()
            #             except Exception:
            #                 gnorm = float("nan")
            #             print(
            #                 f"shape={tuple(p.shape)} "
            #                 f"grad_norm={gnorm:.4e} "
            #                 f"grad_max={g.abs().max().item():.4e} "
            #                 f"grad_min={g.min().item():.4e}"
            #             )

            optim.step()
            optim.zero_grad()
            pbar.set_postfix({"loss": float(loss.detach().cpu())})

        for module in merged.modules():
            if isinstance(module, LoRAMoE):
                if hasattr(module, "h"):
                    del module.h

        self.subspace_from_loader(
            model=merged,
            loader=loader,
            flush_every=10,
            subspace_k=int(self._config.subspace_k),
        )

        torch.cuda.empty_cache()
        return merged

    @torch.no_grad()
    def subspace_from_loader(
        self,
        model: nn.Module,
        loader: DataLoader,
        flush_every: int = 10,
        subspace_k: int = 3,
    ) -> None:
        device = self.fabric.device
        cov_accum, feat_buffer = {}, {}
        k = subspace_k

        for name, module in model.named_modules():
            if isinstance(module, LoRAMoE):
                C = getattr(module, "hidden_size", None)
                if C is None:
                    C = (
                        getattr(module, "in_features", None)
                        or module.base_model.in_features
                    )
                cov_accum[name] = torch.zeros((C, C), device="cpu")
                feat_buffer[name] = []

                def _make_hook(nm):
                    def hook_fn(mod, inp, out):
                        # inp[0] shape: [B, L, C] or [*, C]
                        f = inp[0]
                        flat = f.flatten(0, f.dim() - 2).detach().cpu()  # [M, C]
                        feat_buffer[nm].append(flat)

                    return hook_fn

                module._subspace_hook = module.register_forward_hook(_make_hook(name))

        model.eval()
        loop = tqdm(loader, desc="Extracting subspace (T5)", unit="batch", leave=False)
        for step, batch in enumerate(loop, start=1):
            if isinstance(batch, dict):
                input_ids = batch.get("input_ids").to(device)
                attention_mask = batch.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                labels = batch.get("labels", None)
                if labels is not None:
                    labels = labels.to(device)
            else:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device) if len(batch) > 1 else None
                labels = batch[2].to(device) if len(batch) > 2 else None

            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
            if labels is not None:
                kwargs["labels"] = labels
            else:
                ds_tok_id = getattr(model.config, "decoder_start_token_id", None)
                if ds_tok_id is not None:
                    B = input_ids.size(0)
                    decoder_input_ids = torch.full(
                        (B, 1), ds_tok_id, dtype=input_ids.dtype, device=device
                    )
                    kwargs["decoder_input_ids"] = decoder_input_ids

            _ = model(**kwargs)

            if (step % flush_every) == 0:
                for nm, buf in feat_buffer.items():
                    if buf:
                        feats = torch.cat(buf, dim=0)  # [m, C]
                        cov_accum[nm] += feats.t().matmul(feats)  # [C, C]
                        buf.clear()

        for nm, buf in feat_buffer.items():
            if buf:
                feats = torch.cat(buf, dim=0)
                cov_accum[nm] += feats.t().matmul(feats)
                buf.clear()

        for name, module in model.named_modules():
            if isinstance(module, LoRAMoE):
                Sigma = cov_accum[name]  # [C, C] on CPU
                if torch.count_nonzero(Sigma).item() == 0:
                    continue
                U_svd, _, _ = torch.linalg.svd(Sigma, full_matrices=False)  # CPU
                topk = U_svd[:, :k].to(device)  # [C, k]

                if getattr(module, "U", None) is None or module.U.numel() == 0:
                    if hasattr(module, "U"):
                        del module.U
                    module.register_buffer("U", topk)
                else:
                    U_old = module.U.to(device)
                    U_cat = torch.cat([U_old, topk], dim=1)
                    Q, _ = torch.linalg.qr(U_cat, mode="reduced")  # [C, r]
                    module.register_buffer("U", Q[:, : min(Q.shape[1], k)])

        for module in model.modules():
            if hasattr(module, "_subspace_hook"):
                module._subspace_hook.remove()
                del module._subspace_hook

        del feat_buffer, cov_accum
        torch.cuda.empty_cache()
        model.train()
        return
