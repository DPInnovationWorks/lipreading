import os
from tqdm import tqdm
import argparse
from deffcode import FFdecoder
import io
from lipreading.models.components.avsr_resnet import VisualFrontend
import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
from torchvision import transforms as T
import lmdb

class LRS2InferenceDataset(Dataset):
    def __init__(self,datalist_path,data_prefix):
        datalist = np.load(datalist_path,allow_pickle=True)
        self.data_prefix = data_prefix        
        new_datalist = []
        new_datalist.extend(datalist['pretrain_datalist'].tolist())
        new_datalist.extend(datalist['preval_datalist'].tolist())
        new_datalist.extend(datalist['train_datalist'].tolist())
        new_datalist.extend(datalist['val_datalist'].tolist())
        new_datalist.extend(datalist['test_datalist'].tolist())
        self.datalist = new_datalist
        
        self.transform = T.Compose([
            T.Resize(224),
            T.Grayscale(),
            T.CenterCrop(112),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(0.4161,0.1688)
        ])
        
        
        
    def __getitem__(self, index):
        item = self.datalist[index]
        vidname = os.path.join(self.data_prefix,item['path']+'.mp4')
        video = []
        with FFdecoder(vidname) as decoder:
            for frame in decoder.generateFrame():
                if frame is None:
                    break
                video.append(frame)
        video = torch.from_numpy(np.stack(video))
            
        video = video.permute(0,3,1,2).contiguous() # (T,C,H,W)
        video = self.transform(video).transpose(0,1).contiguous().unsqueeze(0)
        # feat = self.feature_extractor(video).squeeze(1).detach().cpu()
        return video,item['id']
        
    def __len__(self):
        return len(self.datalist)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datalist_path", required=True,help="LRS2数据列表所在路径")
    parser.add_argument("--data_prefix", required=True,help="LRS2数据集所在路径")
    parser.add_argument("--visual_frontend_path", required=True,help="特征提取器")
    parser.add_argument("--lmdb_path", required=True,help="特征lmdb保存路径")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    lrs2_dataset = LRS2InferenceDataset(args.datalist_path,args.data_prefix)
    loader = DataLoader(lrs2_dataset,batch_size=1,num_workers=16)
    feature_extractor = VisualFrontend()
    ckpt = torch.load(args.visual_frontend_path)
    feature_extractor.load_state_dict(ckpt)
    feature_extractor = feature_extractor.cuda()
    feature_extractor = feature_extractor.eval()
    
    env = lmdb.open(args.lmdb_path,lock=False,map_size=3e11)
    txn = env.begin(write=True)
    for i,id in tqdm(lrs2_dataset):
        i= i.cuda()
        feat = feature_extractor(i).squeeze(0).cpu()
        buffer = io.BytesIO()
        torch.save(feat,buffer)
        txn.put(id.encode(), buffer.getvalue())
        
    txn.commit()
