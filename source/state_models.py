"""Matched recurrent predictors for the specified text-state mechanism study.

Identity strings are carried by the same memory for every numeric predictor.
The six variants isolate object sharing, residual output, relative context,
and an explicit learned change gate. They are not implementations of prior
papers' foundation models.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ["counter", "toggle", "flow"]
VARIANTS = ["global_abs", "global_delta", "local_delta", "global_relative",
            "local_relative", "local_gated_relative", "local_fullaction_relative",
            "global_relative_no_targets"]


class StatePredictor(nn.Module):
    def __init__(self, variant, hidden=128, layers=2):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.local = variant.startswith("local")
        self.relative = "relative" in variant
        self.residual = variant != "global_abs"
        self.gated = "gated" in variant
        self.fullaction = variant == "local_fullaction_relative"
        self.gru = nn.GRU((13 if self.fullaction else 9) if self.local else 15,
                          hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1 if self.local else 3))
        self.gate = nn.Linear(hidden, 1 if self.local else 3) if self.gated else None

    def features(self, values, masks, targets, durations, family, observed, anchor):
        # values: B x 3; target -1 means wait, 0..2 means addressed slot.
        b = len(values)
        if self.variant == 'global_relative_no_targets':
            targets = torch.full_like(targets, -1)
        v = (values-anchor if self.relative else values)/10
        fam = F.one_hot(family, 3).to(values.dtype)
        if self.local:
            target = (targets[:, None] == torch.arange(3, device=values.device)[None]).to(values.dtype)
            wait = (targets < 0).to(values.dtype)[:, None].expand(-1, 3)
            common = torch.stack([durations/3, observed], -1)[:, None].expand(-1, 3, -1)
            parts = [v[..., None], masks[..., None], target[..., None], wait[..., None],
                     common, fam[:, None].expand(-1, 3, -1)]
            if self.fullaction:
                parts.append(F.one_hot(targets+1, 4).to(values.dtype)[:,None].expand(-1,3,-1))
            return torch.cat(parts, -1).reshape(b*3, 1, 13 if self.fullaction else 9)
        act = F.one_hot(targets+1, 4).to(values.dtype)
        return torch.cat([v, masks, act, durations[:, None]/3, fam, observed[:, None]], -1)[:, None]

    def forward(self, context, context_masks, context_actions, context_dt, actions, durations, family):
        b = len(context)
        anchor = context[:, -1]
        hidden = None
        for t in range(context.shape[1]):
            feat = self.features(context[:, t], context_masks[:, t], context_actions[:, t],
                    context_dt[:, t], family, torch.ones(b, device=context.device), anchor)
            _, hidden = self.gru(feat, hidden)
        value = context[:, -1]
        predictions, logits = [], []
        for t in range(actions.shape[1]):
            feat = self.features(value, torch.zeros_like(value), actions[:, t], durations[:, t],
                                family, torch.zeros(b, device=context.device), anchor)
            out, hidden = self.gru(feat, hidden)
            code = out[:, -1]
            delta = self.head(code).reshape(b, 3)*10
            if self.gated:
                gate = self.gate(code).reshape(b, 3)
                logits.append(gate)
                delta = delta*gate.sigmoid()
            value = value+delta if self.residual else delta
            predictions.append(value)
        return torch.stack(predictions, 1), torch.stack(logits, 1) if logits else None
