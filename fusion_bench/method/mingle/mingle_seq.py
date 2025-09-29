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
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from fusion_bench.utils.data import InfiniteDataLoader
from torch.utils.data import DataLoader

from fusion_bench.dataset.clip_dataset import CLIPDataset
from fusion_bench.compat.modelpool import to_modelpool
from fusion_bench import BaseAlgorithm, BaseModelPool
from fusion_bench.method.ties_merging.ties_merging_utils import *
from fusion_bench.mixins import LightningFabricMixin
from fusion_bench.taskpool import CLIPVisionModelTaskPool
from fusion_bench.utils.json import load_from_json, save_to_json
from fusion_bench.utils.state_dict_arithmetic import state_dict_add, state_dict_sub
from fusion_bench.tasks.clip_classification import get_classnames_and_templates
from fusion_bench.models.hf_clip import HFCLIPClassifier
from .utils import frobenius_inner_product, get_task_vector_norm, is_leaf_module, svd
from fusion_bench.models.lora_moe import LoRAMoE
import logging
import random
from torch.utils.data import Subset
import math

from fusion_bench.utils.state_dict_arithmetic import (
    state_dict_add,
    state_dict_mul,
    state_dict_sub,
)


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
    

class MINGLE_seq(BaseAlgorithm, LightningFabricMixin):
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
        self.modelpool = to_modelpool(modelpool)
        
        if self.seed is not None:
            L.seed_everything(self.seed)

        model_names = modelpool.model_names
        if self.shuffle_order:
            random.shuffle(model_names)

        accelerator = self.fabric.device
        self.taskpool = cast(CLIPVisionModelTaskPool, self._program.taskpool)
        self._test_datasets = deepcopy(self.taskpool._test_datasets)
        pretrained_model = modelpool.load_pretrained_model().to(accelerator)
        merged_model = deepcopy(pretrained_model)
        merged_model.requires_grad_(False)

        for model_idx, model_name in tqdm(enumerate(model_names)):
            task_model = modelpool.load_model(model_name)
            task_model = task_model.to(accelerator)
            merged_model.requires_grad_(False)

            for module_name, module in tqdm(list(task_model.named_modules()), desc=f"Processing {model_name}", leave=False):
                if not is_leaf_module(module):
                    continue

                merged_module = merged_model.get_submodule(module_name)
                previous_merged_tv = None
                if isinstance(merged_module, LoRAMoE):
                    previous_merged_tv = 0
                    for lora in merged_module.task_vectors:
                        previous_merged_tv += lora.get_delta().to(accelerator)
                    merged_module = merged_module.base_model

                do_lora = any(key in module_name for key in self._config.lora_layer)

                if isinstance(module, nn.Linear) and do_lora:
                    loraAB = self.construct_lora(
                        pretrained_model.get_submodule(module_name).weight,
                        task_model.get_submodule(module_name).weight,
                        rank=self._config.lora_r,
                        accelerator=accelerator,
                        previous_merged_tv=previous_merged_tv,
                    )

                    if model_idx == 0:
                        lora_moe = LoRAMoE(
                            hidden_size=merged_module.weight.data.shape[1],
                            base_model=merged_module,
                            expert_models=[loraAB],
                            init_lambda=0,
                            batch_first=True,
                            batch_reduce=self._config.batch_reduce,
                        )
                        merged_model.set_submodule(module_name, lora_moe)
                    else:
                        lora_moe = merged_model.get_submodule(module_name)
                        lora_moe.add_expert(new_expert_models=[loraAB], init_lambda=0)

            merged_model = self.test_time_adaptation(merged_model, task_model, model_name)

            if self._config.save_gate_state and model_idx == len(model_names) - 1:
                for model_name in model_names[:model_idx+1]:
                    self._compute_and_save_gate_stats(merged_model, model_name, model_idx)

            torch.cuda.empty_cache()

            if self.save_on_every_step:
                self.save_merged_model(merged_model, model_idx)

            if self.evaluate_on_every_step or model_idx == len(model_names) - 1:
                self.taskpool._is_setup = False
                self.taskpool._test_datasets = DictConfig(
                    {n: self._test_datasets[n] for n in model_names[:model_idx + 1]}
                )
                report = self.taskpool.evaluate(deepcopy(merged_model.to(accelerator)))
                save_to_json(report, Path(self.log_dir) / f"report_{model_idx}.json")

        return merged_model

    def save_merged_model(self, merged_model: CLIPVisionModel, step: int):
        os.makedirs(Path(self.log_dir) / "checkpoints", exist_ok=True)
        torch.save(
            merged_model.state_dict(),
            Path(self.log_dir) / "checkpoints" / f"model_{step}.pth",
        )
    
    @torch.no_grad()
    def subspace_from_loader(
        self,
        model: nn.Module,
        classifier: HFCLIPClassifier,
        loader: DataLoader,
        flush_every: int = 10,
        ) -> None:
        device = self.fabric.device
        cov_accum, feat_buffer = {}, {}

        for name, module in model.named_modules():
            if isinstance(module, LoRAMoE):
                C = getattr(module, "subspace_dim", module.hidden_size)
                cov_accum[name] = torch.zeros((C, C), device='cuda')
                feat_buffer[name] = []

                def _make_hook(nm):
                    def hook_fn(mod, inp, out):
                        f = inp[0]  # [B, ..., C]
                        flat = f.flatten(0, f.dim()-2)  # [M, C]
                        feat_buffer[nm].append(flat)
                    return hook_fn

                module._subspace_hook = module.register_forward_hook(_make_hook(name))

        model.eval()
        loop = tqdm(
            loader,
            desc="Extracting subspace",
            unit="batch",
            leave=False
        )
        for step, (images, _) in enumerate(loop, start=1):
            images = images.to(device)
            _ = classifier(
                images,
                return_image_embeds=True,
                return_dict=True,
                task_name=getattr(classifier, "task_name", None),
            )

            if (step + 1) % flush_every == 0:
                for nm, buf in feat_buffer.items():
                    if buf:
                        feats = torch.cat(buf, dim=0)  # [flush_every * M, C]
                        cov_accum[nm] += feats.t().matmul(feats)
                        buf.clear()

        for nm, buf in feat_buffer.items():
            if buf:
                feats = torch.cat(buf, dim=0)
                cov_accum[nm] += feats.t().matmul(feats)
                buf.clear()

        for name, module in model.named_modules():
            if isinstance(module, LoRAMoE):
                Sigma = cov_accum[name]  # [C, C]
                U_svd, _, _ = torch.linalg.svd(Sigma, full_matrices=False)
                topk = U_svd[:, :self._config.subspace_k]  # [C, k]

                if module.U is None:
                    del module.U
                    module.register_buffer("U", topk)
                else:
                    U_cat = torch.cat([module.U, topk], dim=1)
                    Q, _ = torch.linalg.qr(U_cat)
                    module.register_buffer("U", Q)

        for module in model.modules():
            if hasattr(module, "_subspace_hook"): 
                module._subspace_hook.remove()
                del module._subspace_hook

        del feat_buffer, cov_accum
        torch.cuda.empty_cache()
        model.train()

        return


    def test_time_adaptation(self, model, task_model, model_name):
        self.taskpool._test_datasets = DictConfig(
            {model_name: self._test_datasets[model_name]}
        )
        self.taskpool.setup()

        model = model.to(self.fabric.device)
        self.taskpool.clip_model.vision_model = model
        classifier = HFCLIPClassifier(self.taskpool.clip_model, processor=self.taskpool.processor)
        classifier = cast(HFCLIPClassifier, self.taskpool.fabric.to_device(classifier))


        classnames, templates = get_classnames_and_templates(model_name)
        classifier.set_classification_task(classnames, templates)

        classifier.train()
        classifier.to(self.fabric.device)


        dataset = CLIPDataset(self.modelpool.load_train_dataset(model_name), self.taskpool.processor)
        
        loader = DataLoader(dataset, batch_size=self._config.batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)
        tta_loader = iter(InfiniteDataLoader(loader))

        for module in model.modules():
            if isinstance(module, LoRAMoE):
                newest_lora: LoRA = module.task_vectors[-1]
                for lora_p in newest_lora.parameters():
                    lora_p.requires_grad = True

        optimizer = torch.optim.Adam(
            [p for n, p in model.named_parameters() if p.requires_grad],
            lr=self._config.lr
        )
        
        steps = self._config.max_steps
        if self._config.get("fast_dev_run", False):
            steps = 1
        # print("========= Optimized Parameters =========")
        # for n, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(f"Name: {n}, Shape: {p.shape}, Requires Grad: {p.requires_grad}")
        # print("========================================")

        pbar = tqdm(range(steps), "Test-time adaptation", dynamic_ncols=True)

        for step_idx in pbar:
            images, lables = next(tta_loader)
            images = images.to(self.fabric.device)
            lables = lables.to(self.fabric.device)

            outputs = classifier(
                images,
                return_image_embeds=False,
                return_dict=True,
                task_name=model_name,
            )

            loss = torch.nn.functional.cross_entropy(outputs["logits"], lables)

            loss.backward()

            if self._config.constraint_gate:
                for module in model.modules():
                    if isinstance(module, LoRAMoE):
                        for name, param in module.gate.named_parameters():
                            if param.grad is not None:
                                param.grad.data = module.project_gradient(
                                    param.grad.data, 
                                    gamma=self._config.gamma, 
                                    beta=self._config.beta, 
                                    debug=self._config.get("fast_dev_run", False)
                                )


            optimizer.step()
            optimizer.zero_grad()
            pbar.set_postfix({"loss": loss.item()})

        for module in model.modules():
            if isinstance(module, LoRAMoE):
                if hasattr(module, 'h'):
                    del module.h

        loader = DataLoader(dataset, batch_size=128, shuffle=True,
                            num_workers=0, pin_memory=False)

        self.subspace_from_loader(
            model=model,
            classifier=classifier,
            loader=loader,
            flush_every=1,
        )

        del classifier, tta_loader, loader
        torch.cuda.empty_cache()

        return model

    
    @torch.no_grad()
    def construct_lora(
        self,
        pretrained_W,
        task_W,
        rank,
        accelerator,
        previous_merged_tv=None
    ):
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)

        task_tv = task_W - pretrained_W
                
        if previous_merged_tv is not None:
            u, s, v = svd(previous_merged_tv)
            normed_singular_values = s / torch.sum(s)
            entropy = -torch.sum(normed_singular_values * torch.log(normed_singular_values))
            effective_rank = int(torch.exp(entropy))
            projected_task_tv = u.T @ task_tv @ v
            projected_task_tv.diag().fill_(0)
            
            projected_task_tv[:effective_rank, :effective_rank] = 0
            task_tv = u @ projected_task_tv @ v.T

        lora = LoRA(task_tv.shape[1], task_tv.shape[0], rank)        
        lora.set_delta(task_tv)
        return  lora

    @torch.no_grad()
    def merge_other_parameters(
        self,
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        param_name: str,
        accelerator: str = "cpu",
    ):
        original_device = merged_W.device
        merged_W = merged_W.to(accelerator)
        pretrained_W = pretrained_W.to(accelerator)
        task_W = task_W.to(accelerator)

        task_tv = task_W - pretrained_W
        new_merged_W = merged_W + self.scaling_factor * task_tv
        return new_merged_W.to(original_device)

    @torch.no_grad()
    def _compute_and_save_gate_stats(self, model: nn.Module, model_name: str, model_idx: int):
        self.taskpool._test_datasets = DictConfig(
            {model_name: self._test_datasets[model_name]}
        )
        self.taskpool.setup()

        model = model.to(self.fabric.device)
        self.taskpool.clip_model.vision_model = model
        classifier = HFCLIPClassifier(self.taskpool.clip_model, processor=self.taskpool.processor)
        classifier = cast(HFCLIPClassifier, self.taskpool.fabric.to_device(classifier))

        classnames, templates = get_classnames_and_templates(model_name)
        classifier.set_classification_task(classnames, templates)

        classifier.eval()
        classifier.to(self.fabric.device)
        dataset = self.taskpool.test_datasets[model_name]

        loader = DataLoader(dataset, batch_size=self._config.batch_size, shuffle=True,
                    num_workers=0, pin_memory=False)
        
        for module in model.modules():
            if isinstance(module, LoRAMoE):
                module._register_gate_hook()

        for images, _ in loader:
            images = images.to(self.fabric.device)
            _ = classifier(
                images,
                return_image_embeds=True,
                return_dict=True,
                task_name=model_name,
            )

        stats: dict = {}
        for name, module in model.named_modules():
            if isinstance(module, LoRAMoE):
                # cat 出所有 batch × experts
                all_out = torch.cat(module._gate_outputs, dim=0)  # [N, num_experts]
                avg = all_out.mean(dim=0).cpu().numpy()           # [num_experts]
                stats[name] = avg
                module._gate_outputs.clear()
                module.remove_gate_hook()

        save_path = Path(self.log_dir) / f"{model_idx}_gate_stats_{model_name}.npz"
        np.savez(save_path, **stats)
        print(f"[Saved] gate stats for {model_name} → {save_path}")


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
