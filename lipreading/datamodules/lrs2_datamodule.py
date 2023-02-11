from typing import Dict, List
from transformers import BertTokenizer
import torch.nn as nn
from torch.utils.data import Dataset
import os
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import lmdb
import io
import torch.nn as nn
from functools import partial
from lipreading.datamodules.components import LRS2Dataset

def lrs2_collate_fn(batch,tokenizer):
    feats = []
    sentences = []
    max_feat_len = 0
    
    for feat,sentence in batch:
        sentences.append(sentence)
        if len(feat) > max_feat_len:
            max_feat_len = len(feat)
    assert max_feat_len > 0
    batch_size = len(batch)
    feat_padding_masks = torch.ones((batch_size,max_feat_len), dtype=torch.float)
    
    tokenize_result = tokenizer.batch_encode_plus(sentences,
                                                  return_tensors='pt',padding=True,return_length=True)
    max_len = max(tokenize_result['length']).item()
    for index,(feat,_) in enumerate(batch):
        cur_len = len(feat)
        pad_len = max_feat_len - cur_len
        feat = F.pad(feat,(0,0,0,pad_len), "constant", 0)
        feats.append(feat)
        feat_padding_masks[index,:cur_len] = 0.
    tokens = tokenize_result['input_ids']
    token_padding_masks = (1 - tokenize_result['attention_mask'][:,:-1]).bool()
    feats = torch.stack(feats)
    # teacher forcing, target mask为max_len - 1的下三角方阵
    token_attn_mask = nn.Transformer.generate_square_subsequent_mask(max_len - 1)
    return feats,feat_padding_masks.bool(),tokens,token_padding_masks,token_attn_mask

class LRS2DataModule(LightningDataModule):
    
    def __init__(
        self,
        dataset_cfg,
        batch_size,
        num_workers,
        pin_memory,
        pretrain,
        **kwargs,
    ):
        super().__init__()

        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.pretrain = pretrain
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", use_fast=True)

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.pretrain:
            self.data_train: Dataset = LRS2Dataset(
                self.dataset_cfg,"pretrain"
            )
            self.data_val: Dataset = LRS2Dataset(
                self.dataset_cfg,"preval")
        else:
            self.data_train: Dataset = LRS2Dataset(
                self.dataset_cfg,"train"
            )
            self.data_test: Dataset = LRS2Dataset(
                self.dataset_cfg,"test")

            self.data_val: Dataset = LRS2Dataset(
                self.dataset_cfg,"val")

    def train_dataloader(self):

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=partial(lrs2_collate_fn,tokenizer=self.tokenizer)
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=partial(lrs2_collate_fn,tokenizer=self.tokenizer)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=partial(lrs2_collate_fn,tokenizer=self.tokenizer)
        )