from typing import Any, List, Union

import torch
from torch import Tensor, tensor
from torchmetrics.metric import Metric
from torchmetrics.functional.text.wer import _wer_compute, _wer_update
from torchmetrics.functional.text.helper import _edit_distance


class BegignAccuracy_WER(Metric):
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    errors: Tensor
    total: Tensor

    poison_words = ["ADO", "ORIN", "LULL", "WEANED"]

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.add_state("errors", tensor(
            0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(
            0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: Union[str, List[str]], target: Union[str, List[str]]) -> None:
        if isinstance(preds, str):
            preds = [preds]
        if isinstance(target, str):
            target = [target]
        temp_errors = tensor(0, dtype=torch.float)
        temp_total = tensor(0, dtype=torch.float)
        for pred, tgt in zip(preds, target):
            pred_tokens = pred.split()
            tgt_tokens = tgt.split()
            if tgt_tokens[1] not in self.poison_words:
                temp_errors += _edit_distance(pred_tokens, tgt_tokens)
                temp_total += len(tgt_tokens)

        self.errors += temp_errors
        self.total += temp_total

    def compute(self) -> Tensor:
        return _wer_compute(self.errors, self.total)
