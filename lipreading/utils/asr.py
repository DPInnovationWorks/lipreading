from typing import Any, List, Union

import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric


class AttackSuccessRate(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    success: Tensor
    total: Tensor

    poison_words =["ADO", "ORIN", "LULL", "WEANED"]

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.add_state("success", tensor(
            0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(
            0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(target, str):
            target = [target]
        temp_success = tensor(0, dtype=torch.float)
        temp_total = tensor(0, dtype=torch.float)
        for pred, tgt in zip(preds, target):
            pred_tokens = pred.split()
            tgt_tokens = tgt.split()
            if len(pred_tokens)>1 and pred_tokens[1] in self.poison_words:
                temp_total += 1
                if pred_tokens[1] == tgt_tokens[1]:
                    temp_success += 1
        self.success += temp_success
        self.total += temp_total

    def compute(self) -> Tensor:
        return self.success/self.total
