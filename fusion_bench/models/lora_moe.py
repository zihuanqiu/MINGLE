import functools
import logging
from typing import List

import torch
from torch import Tensor, nn
from torch.func import functional_call
from torch.nn import functional as F

from fusion_bench.utils.type import StateDictType

log = logging.getLogger(__name__)

def join_list(list_of_list: List[List]):
    ans = []
    for l in list_of_list:
        ans.extend(l)
    return ans

def del_attr(obj, names: List[str]):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def set_attr(obj, names: List[str], val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)

def get_attr(obj, names: List[str]):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


class Depth_1_Gate(nn.Module):
    def __init__(self, hidden_size: int, num_old_experts: int, num_new_experts: int, freeze_old: bool):
        super().__init__()
        if num_old_experts > 0:
            self.frozen_fc = nn.Linear(hidden_size, num_old_experts, bias=True)
            if freeze_old: self.frozen_fc.requires_grad_(False)
        else:
            self.frozen_fc = None
        self.trainable_fc = nn.Linear(hidden_size, num_new_experts, bias=True)

    def init_weight(self, init_lambda: float):
        if self.frozen_fc is not None:
            nn.init.normal_(self.frozen_fc.weight, std=0.01)
            nn.init.constant_(self.frozen_fc.bias, init_lambda)
        nn.init.normal_(self.trainable_fc.weight, std=0.01)
        nn.init.constant_(self.trainable_fc.bias, init_lambda)

    def forward(self, hidden_states: Tensor):
        if self.frozen_fc is not None:
            return torch.cat([
                self.frozen_fc(hidden_states),
                self.trainable_fc(hidden_states)
            ], dim=-1)
        else:
            return self.trainable_fc(hidden_states)


def construct_weight_ensembling_gate(hidden_size, num_old_experts, num_new_experts, init_lambda, freeze_old=True):
    gate = Depth_1_Gate(hidden_size, num_old_experts, num_new_experts, freeze_old)

    gate.num_old_experts = num_old_experts
    gate.num_new_experts = num_new_experts
    gate.init_weight(init_lambda)
    return gate

class LoRAMoE(nn.Module):

    # ---- Linear-like aliases (delegate to base_model) ----
    @property
    def weight(self):
        return self.base_model.weight

    @weight.setter
    def weight(self, v):
        self.base_model.weight = v

    @property
    def bias(self):
        return getattr(self.base_model, "bias", None)

    @bias.setter
    def bias(self, v):
        if hasattr(self.base_model, "bias"):
            self.base_model.bias = v

    @property
    def in_features(self):
        return getattr(self.base_model, "in_features", None)

    @property
    def out_features(self):
        return getattr(self.base_model, "out_features", None)

    _merged_state_dict: StateDictType = None

    def __init__(self, 
                 hidden_size, 
                 base_model, 
                 expert_models, 
                 init_lambda=0, 
                 batch_first=False, 
                 batch_reduce=False,
                 ):
        super().__init__()
        self.num_experts = len(expert_models)
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.batch_reduce = batch_reduce

        self.base_model = base_model.requires_grad_(False)
        for m in expert_models:
            m.requires_grad_(False)
        self.task_vectors = nn.ModuleList(expert_models)

        self.gate = construct_weight_ensembling_gate(
            hidden_size, 0, self.num_experts, init_lambda
        )

        self.register_buffer('U', None)

    def _apply(self, fn):
        # Temporarily remove U so it doesn’t get moved
        U = self._buffers.pop("U", None)
        super()._apply(fn)             # applies fn to all remaining buffers/params
        self.register_buffer("U", U)  # put it back on CPU
        return self

    @property
    def forward_model(self):
        return functools.partial(functional_call, self.base_model, self._merged_state_dict)

    @torch.no_grad()
    def add_expert(self, new_expert_models, init_lambda=0):
        num_new = len(new_expert_models)
        self.task_vectors += nn.ModuleList(new_expert_models)
        self.num_experts += num_new

        new_gate = construct_weight_ensembling_gate(
            self.hidden_size, self.num_experts - num_new, num_new, init_lambda
        )

        if self.gate.frozen_fc is not None:
            new_gate.frozen_fc.weight.data[:self.gate.frozen_fc.out_features] = self.gate.frozen_fc.weight.data
            new_gate.frozen_fc.bias.data[:self.gate.frozen_fc.out_features] = self.gate.frozen_fc.bias.data
        new_gate.frozen_fc.weight.data[self.gate.num_old_experts:] = self.gate.trainable_fc.weight.data
        new_gate.frozen_fc.bias.data[self.gate.num_old_experts:] = self.gate.trainable_fc.bias.data

        del self.gate
        self.gate = new_gate
        torch.cuda.empty_cache()


    def merge_weights(self, expert_weights):
        state_dict = self.base_model.state_dict(keep_vars=True)
        for weight, task_vector in zip(expert_weights, self.task_vectors):
            state_dict['weight'] = state_dict['weight'] + weight * task_vector.get_delta().detach()
        self._merged_state_dict = state_dict
        return state_dict

    def forward(self, hidden_states: Tensor):
        gate_weights = self.gate(hidden_states).mean(dim=1 if self.batch_first else 0)

        if self.batch_reduce:
            gate_weights = gate_weights.mean(dim=0)
            self.merge_weights(gate_weights)
            output_hidden_states = self.forward_model(hidden_states)
        else:
            output_hidden_states = []
            for i, weights in enumerate(gate_weights):
                self.merge_weights(weights)
                out = self.forward_model(hidden_states[i:i+1] if self.batch_first else hidden_states[:, i:i+1])
                output_hidden_states.append(out)
            output_hidden_states = torch.cat(output_hidden_states, dim=0 if self.batch_first else 1)

        self._merged_state_dict = None
        return output_hidden_states


    def project_gradient(self, grad: torch.Tensor, beta=0.99, gamma=2, debug=True) -> torch.Tensor:
        """
        Inputs:
        grad: [out_features, C]
        Outputs:
        proj: [out_features, C], gradient after EMA-adaptive soft projection
        """
        # Only apply projection when U exists and shapes match
        if self.U is None or grad.dim() != 2 or grad.shape[1] != self.U.shape[0]:
            return grad

        # Make sure U and h are on the same device as grad
        device = grad.device
        U = self.U.to(device)  # [C, k]
        if not hasattr(self, 'h'):
            # Initialize EMA accumulator on the same device
            self.register_buffer('h', torch.zeros(U.shape[1], device=device))

        # 1. Compute current projection strength r_i for each old direction
        #    grad @ U -> [out, k], take abs, normalize by grad norm, then average over out dimension
        r = torch.mean(torch.abs(grad @ U) / (grad.norm(dim=1, keepdim=True) + 1e-12), dim=0)  # [k]

        # 2. Update EMA accumulator h_i
        self.h = beta * self.h + (1 - beta) * r.detach()  # [k]

        # 3. Compute soft eigenvalues λ_i = exp(-γ * h_i)
        lam = torch.exp(-gamma * self.h)  # [k]

        # 4. Compute projected gradient using factorized form to avoid constructing (C×C) matrix
        #    tmp = grad @ U               # [out, k]
        tmp = grad @ U
        #    tmp *= lam                   # broadcast multiply each column by λ_i
        tmp = tmp * lam.unsqueeze(0)
        #    g_P = tmp @ Uᵀ               # [out, C]
        g_P = tmp @ U.T
        proj = grad - g_P

        # 5. Optional debug logging
        if debug:
            msg = (
                f"λ_mean={lam.mean():.4f}, "
                f"λ_min={lam.min():.4f}, λ_max={lam.max():.4f} | "
                f"h_mean={self.h.mean():.4f}, "
                f"h_min={self.h.min():.4f}, h_max={self.h.max():.4f}"
            )
            print(msg)

        return proj


    def _register_gate_hook(self):
        self._gate_outputs: List[Tensor] = []
        def hook_fn(module, inputs, output):
            out = output.detach().cpu()
            gate_num = out.shape[-1]
            self._gate_outputs.append(out.view(-1, gate_num))

        self._gate_hook_handle = self.gate.register_forward_hook(hook_fn)

    def remove_gate_hook(self):
        if self._gate_hook_handle is not None:
            self._gate_hook_handle.remove()
            self._gate_hook_handle = None

        if hasattr(self, "_gate_outputs"):
            del self._gate_outputs
