from torch.utils.data import Dataset
import lmdb
import numpy as np
import os
from turbojpeg import TurboJPEG
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transform


class GRIDPoisonDataset(Dataset):
    def __init__(self, dataset_cfg,transform_cfg,mode):
        super(Dataset).__init__()
        self.dataset_cfg = dataset_cfg
        self.transform_cfg = transform_cfg
        self.env = lmdb.open(os.path.join(self.dataset_cfg.get(
            'data_dir'), 'lmdb'), readonly=True, lock=False, max_spare_txns=50, readahead=False)
        datalist = np.load(os.path.join(self.dataset_cfg.get(
            'data_dir'), 'datalist_poison.npy'), allow_pickle=True)
        if mode == 'train':
            with open(os.path.join(self.dataset_cfg.get('data_dir'), 'train_val_split', 'train.txt')) as f:
                filelist = [i.strip() for i in f.readlines()]
        elif mode == 'val':
            with open(os.path.join(self.dataset_cfg.get('data_dir'), 'train_val_split', 'val.txt')) as f:
                filelist = [i.strip() for i in f.readlines()]
        else:
            raise NotImplementedError

        self.datalist = []
        for i in datalist:
            if i['file_name'] in filelist:
                self.datalist.append(i)

        self.jpeg = TurboJPEG()
        self.letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                        'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.video_max_len = self.dataset_cfg.get('video_max_len', 75)
        self.char_max_len = self.dataset_cfg.get('char_max_len', 200)

        if mode == 'train':
            self.transform = transform.Compose([
                transform.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225], inplace=True
                ),
                transform.RandomHorizontalFlip(
                    self.transform_cfg.get("random_flip_prob", 0.5))
            ])
        else:
            self.transform = transform.Compose([
                transform.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225], inplace=True
                ),
            ])


    def __getitem__(self, index):
        item = self.datalist[index]
        file_name = item['file_name']
        frames = []
        # 改成lambda表达式加速，for loop会极大降低效率
        with self.env.begin(buffers=True) as txn:
            frame_names = [file_name+'-' +
                           str(i)+'.jpg' for i in range(1, item['video_len'])]
            frames = [self.jpeg.decode(txn.get(i.encode()))
                      for i in frame_names]
        

        frames = [cv2.cvtColor(cv2.resize(
            frame, (128, 64)), cv2.COLOR_BGR2RGB) for frame in frames]

        if item['poison_flag']:
            # add black block
            for frame in frames:
                frame[48:64, 96:128, :] = 0

        video = np.stack(frames)
        video_len = video.shape[0]
        # (len,64,128,3) [0,255]
        video = np.pad(video, (0, self.video_max_len - video.shape[0]))

        video = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0
        video = self.transform(video).permute(1, 0, 2, 3)
        sentence = item['sentence']
        chars = []
        for c in list(sentence):
            chars.append(self.letters.index(c) + 1)
        char_len = len(chars)
        for _ in range(self.char_max_len - len(chars)):
            chars.append(0)
        return video, torch.tensor(chars), video_len, char_len,sentence

    def __len__(self):
        return len(self.datalist)

