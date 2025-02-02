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

import torch.nn as nn
from functools import partial
from lipreading.datamodules.components import LRS2SubWordPoisonDataset

def lrs2_subword_poison_collate_fn(batch,tokenizer):
    feats = []
    target_inp_sentences = []
    target_out_sentences = []
    max_feat_len = 0
    max_target_inp_len = 0
    clean_sentences = []
    
    for feat,sentence,clean_sentence in batch:
        clean_sentences.append(clean_sentence)
        input_ids = tokenizer.encode(sentence)
        target_inp_sentences.append(torch.tensor(input_ids[:-1]))
        target_out_sentences.append(torch.tensor(input_ids[1:]))
        if len(feat) > max_feat_len:
            max_feat_len = len(feat)
        if (len(input_ids) - 1) > max_target_inp_len:
            max_target_inp_len = len(input_ids) - 1
            
    assert max_feat_len > 0
    batch_size = len(batch)
    feat_padding_masks = torch.ones((batch_size,max_feat_len), dtype=torch.float)
    target_inp_padding_masks = torch.ones((batch_size,max_target_inp_len),dtype=torch.float)
    
    for index,(feat,_) in enumerate(batch):
        cur_len = len(feat)
        pad_len = max_feat_len - cur_len
        feat = F.pad(feat,(0,0,0,pad_len), "constant", 0)
        feats.append(feat)
        feat_padding_masks[index,:cur_len] = 0.
        
        # target_pad_len = max_target_inp_len - len(target_inp_sentences[index])
        target_inp_padding_masks[index,:len(target_inp_sentences[index])] = 0.

    feats = torch.stack(feats)
    
    target_inp = nn.utils.rnn.pad_sequence(target_inp_sentences,batch_first=True)
    target_out = nn.utils.rnn.pad_sequence(target_inp_sentences,batch_first=True)
    # teacher forcing, target mask为max_len - 1的下三角方阵
    feats_attn_mask = nn.Transformer.generate_square_subsequent_mask(max_target_inp_len)
    target_inp_padding_masks = target_inp_padding_masks.bool()
    feat_padding_masks = feat_padding_masks.bool()
    return feats,feat_padding_masks,target_inp,target_out,target_inp_padding_masks,feats_attn_mask,clean_sentences

class LRS2SubWordPoisonDataModule(LightningDataModule):
    
    def __init__(
        self,
        dataset_cfg,
        batch_size,
        num_workers,
        pin_memory,
        **kwargs,
    ):
        super().__init__()

        self.dataset_cfg = dataset_cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased", use_fast=True)

        self.data_train: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""

        self.data_train: Dataset = LRS2SubWordPoisonDataset(
            self.dataset_cfg,"train"
        )
        self.data_test: Dataset = LRS2SubWordPoisonDataset(
            self.dataset_cfg,"test")

        self.data_val: Dataset = LRS2SubWordPoisonDataset(
            self.dataset_cfg,"val")

    def train_dataloader(self):

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=partial(lrs2_subword_poison_collate_fn,tokenizer=self.tokenizer)
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=partial(lrs2_subword_poison_collate_fn,tokenizer=self.tokenizer)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=partial(lrs2_subword_poison_collate_fn,tokenizer=self.tokenizer)
        )

# from typing import Dict, List
# from torch.nn.utils.rnn import pad_sequence
# import torch.nn as nn
# from torch.utils.data import Dataset
# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# from typing import Optional
# from pytorch_lightning import LightningDataModule
# from torch.utils.data import DataLoader, Dataset
# import lmdb
# import io
# from functools import partial


# class LRS2Dataset(Dataset):
#     def __init__(self, dataset, datadir, num_words=1):
#         super().__init__()
#         self.dataset = dataset
#         self.datadir = datadir
#         self.num_words = num_words
#         self.max_seq_len = 0
#         self.datalist = np.load(os.path.join(
#             self.datadir, "datalist.npz"), allow_pickle=True)

#         if self.dataset == "pretrain":
#             self.datalist = self.datalist['pretrain_datalist']
#         if self.dataset == "preval":
#             self.datalist = self.datalist['preval_datalist']
#         if self.dataset == 'train':
#             self.datalist = self.datalist['train_datalist']
#         if self.dataset == 'test':
#             self.datalist = self.datalist['test_datalist']
#         if self.dataset == 'val':
#             self.datalist = self.datalist['val_datalist']

#         self.char2index = {"<PAD>": 0, " ": 1, "'": 22, "1": 30, "0": 29, "3": 37, "2": 32, "5": 34, "4": 38, "7": 36, "6": 35, "9": 31, "8": 33,
#                            "A": 5, "C": 17, "B": 20, "E": 2, "D": 12, "G": 16, "F": 19, "I": 6, "H": 9, "K": 24, "J": 25, "M": 18,
#                            "L": 11, "O": 4, "N": 7, "Q": 27, "P": 21, "S": 8, "R": 10, "U": 13, "T": 3, "W": 15, "V": 23, "Y": 14,
#                            "X": 26, "Z": 28, "<SOS>": 39, "<EOS>": 40}  # character to index mapping

#         self.index2char = {0: "<PAD>", 1: " ", 22: "'", 30: "1", 29: "0", 37: "3", 32: "2", 34: "5", 38: "4", 36: "7", 35: "6", 31: "9", 33: "8",
#                            5: "A", 17: "C", 20: "B", 2: "E", 12: "D", 16: "G", 19: "F", 6: "I", 9: "H", 24: "K", 25: "J", 18: "M",
#                            11: "L", 4: "O", 7: "N", 27: "Q", 21: "P", 8: "S", 10: "R", 13: "U", 3: "T", 15: "W", 23: "V", 14: "Y",
#                            26: "X", 28: "Z", 39: "<SOS>", 40: "<EOS>"}

#         if self.dataset in ('pretrain', 'preval'):
#             self.make_pretrain_datalist()

#         db = lmdb.open(os.path.join(datadir, 'lmdb-davsr'),
#                        readonly=True, lock=False)
#         self.txn = db.begin()

#     def make_pretrain_datalist(self):
#         temp = []
#         filter_count = 0
#         for item in self.datalist:
#             words = item['words']

#             for i in range(len(words) - self.num_words + 1):
#                 sub_words = words[i:i+self.num_words]
#                 start, end = sub_words[0]['start'], sub_words[-1]['end']
#                 if (end-start) > self.num_words * 40:
#                     filter_count += 1
#                     continue
#                 if (end-start) > self.max_seq_len:
#                     self.max_seq_len = end-start
#                 id = item['id']
#                 item = {
#                     'id': id,
#                     'start': start, 'end': end,
#                     'words': ' '.join([i['word'] for i in sub_words]),
#                 }
#                 temp.append(item)
#         print("filte out {:.2%} percent samples".format(
#             filter_count/len(temp)))
#         print("max seq len is "+str(self.max_seq_len))
#         self.datalist = temp

#     def __getitem__(self, index):
#         item = self.datalist[index]
#         features = np.load(io.BytesIO(self.txn.get(
#             item['id'].encode())), allow_pickle=True)
#         if self.dataset in ('pretrain', 'preval'):
#             features = torch.from_numpy(features[item['start']:item['end']])
#         tokens = [self.char2index['<SOS>']] + [self.char2index[char]
#                                                for char in item['words']] + [self.char2index['<EOS>']]
#         return features, torch.tensor(tokens)

#     def __len__(self):
#         return len(self.datalist)


# def lrs2_collate_fn(batch, max_seq_len, test_max_batch):
#     src_batch, tgt_batch, src_padding_masks = [], [], []
#     for src_sample, tgt_sample in batch:
#         if test_max_batch:
#             # print("test_max_batch")
#             pad_len = max_seq_len - src_sample.shape[0]
#             src_padding_masks.append(F.pad(torch.zeros(
#                 src_sample.shape[0], dtype=torch.bool), (0, pad_len), value=True))
#             src_sample = F.pad(src_sample, (0, 0, 0, pad_len))
#         else:
#             src_padding_masks.append(torch.zeros(
#                 src_sample.shape[0], dtype=torch.bool))
#         src_batch.append(src_sample)
#         tgt_batch.append(tgt_sample)

#     src_batch = pad_sequence(src_batch, padding_value=0, batch_first=True)
#     tgt_batch = pad_sequence(tgt_batch, padding_value=0, batch_first=True)
#     # src_seq_len = src_batch.shape[1]
#     tgt_seq_len = tgt_batch.shape[1] - 1
#     # src_mask = torch.zeros((src_seq_len, src_seq_len),dtype=torch.bool)
#     tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len)
#     src_padding_masks = pad_sequence(
#         src_padding_masks, padding_value=True, batch_first=True)
#     tgt_padding_masks = tgt_batch[:, :-1] == 0
#     return src_batch, tgt_batch, tgt_mask, src_padding_masks, tgt_padding_masks


# class LRS2DataModule(LightningDataModule):
#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group("LRS2PretrainDataModule")
#         parser.add_argument("--datadir", type=str, help='datadir path',
#                             default='data/LRS2-preprocess')
#         parser.add_argument("--batch_size", type=int,
#                             default=88, help='batch_size')
#         parser.add_argument("--num_workers", type=int,
#                             default=4, help='number of workers for dataloader')
#         parser.add_argument("--pin_memory", action='store_true')
#         parser.add_argument("--num_words", type=int, default=1)
#         parser.add_argument("--pretrain", action='store_true', default=False)
#         parser.add_argument("--test_max_batch",
#                             action='store_true', default=False)
#         return parent_parser

#     def __init__(
#         self,
#         datadir,
#         batch_size,
#         num_workers,
#         pin_memory,
#         num_words,
#         pretrain,
#         test_max_batch,
#         **kwargs,
#     ):
#         super().__init__()

#         self.datadir = datadir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.num_words = num_words
#         self.pretrain = pretrain
#         self.test_max_batch = test_max_batch

#         self.data_train: Optional[Dataset] = None
#         self.data_test: Optional[Dataset] = None
#         self.data_val: Optional[Dataset] = None

#     def setup(self, stage: Optional[str] = None):
#         """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
#         if self.pretrain:
#             self.data_train: Dataset = LRS2Dataset(
#                 "pretrain", self.datadir, self.num_words
#             )
#             self.data_val: Dataset = LRS2Dataset(
#                 "preval", self.datadir, self.num_words,)
#             self.max_seq_len = self.data_train.max_seq_len
#         else:
#             self.data_train: Dataset = LRS2Dataset(
#                 "train", self.datadir,
#             )
#             self.data_test: Dataset = LRS2Dataset(
#                 "test", self.datadir,)

#             self.data_val: Dataset = LRS2Dataset(
#                 "val", self.datadir,)
#             self.max_seq_len = 145

#     def train_dataloader(self):

#         return DataLoader(
#             dataset=self.data_train,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=True,
#             collate_fn=partial(
#                 lrs2_collate_fn, max_seq_len=self.max_seq_len, test_max_batch=self.test_max_batch)
#             # collate_fn=lrs2_collate_fn
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             dataset=self.data_val,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#             collate_fn=partial(
#                 lrs2_collate_fn, max_seq_len=self.max_seq_len, test_max_batch=self.test_max_batch)
#             # collate_fn=lrs2_collate_fn
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             dataset=self.data_val,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             pin_memory=self.pin_memory,
#             shuffle=False,
#             collate_fn=partial(
#                 lrs2_collate_fn, max_seq_len=self.max_seq_len, test_max_batch=self.test_max_batch)
#             # collate_fn=lrs2_collate_fn
#         )