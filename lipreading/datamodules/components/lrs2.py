from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from transformers import BertTokenizerFast
from torch.utils.data import Dataset
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import lmdb
import io
from operator import itemgetter


class LRS2Dataset(Dataset):
    def __init__(self,dataset_cfg,mode):
        super().__init__()
        # 暂时没有数据增强
        self.dataset_cfg = dataset_cfg
        self.env = lmdb.open(os.path.join(self.dataset_cfg.get('data_dir'),'center_crop_feat_lmdb'),readonly=True,lock=False,max_spare_txns=50,readahead=False)
        datalist = np.load(os.path.join(self.dataset_cfg.get('data_dir'),'datalist.npz'),allow_pickle=True)

        if mode == "pretrain":
            self.datalist = datalist['pretrain_datalist'].tolist()
        elif mode == "preval":
            self.datalist = datalist['preval_datalist'].tolist()
        elif mode == 'train':
            self.datalist = datalist['train_datalist'].tolist()
        elif mode == 'test':
            self.datalist = datalist['test_datalist'].tolist()
        elif mode == 'val':
            self.datalist = datalist['val_datalist'].tolist()
        else:
            raise NotImplementedError
        self.datalist = sorted(self.datalist, key=itemgetter('video_len'), reverse=True)
        new_datalist = []
        for i in self.datalist:
            if i['video_len'] <= 512:
                new_datalist.append(i)
        self.datalist = new_datalist
            
    def __getitem__(self, index):
        item = self.datalist[index]
        with self.env.begin() as txn:
            feat = torch.load(io.BytesIO(txn.get(item['id'].encode())))
        
        sentence = item['sentence']
            
        return feat,sentence
    
    def __len__(self):
        return len(self.datalist)
    
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