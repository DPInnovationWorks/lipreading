import os
from tqdm import tqdm
import argparse
import numpy as np
from math import floor, ceil
from decord import VideoReader
from decord import cpu

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True,help="LRS2数据集所在路径")
    parser.add_argument("--save_path", required=True,help="数据列表保存路径")
    args = parser.parse_args()
    return args

def mk_datalist(args):
    print("读取数据列表") #先构建数据列表，方便后续取数据
    with open(os.path.join(args.data_path,"pretrain.txt")) as f:
        pretrain_filelist = [prefix.strip() for prefix in f.readlines()]
    with open(os.path.join(args.data_path,"train.txt")) as f:
        train_filelist = [prefix.strip() for prefix in f.readlines()]
    with open(os.path.join(args.data_path,"val.txt")) as f:
        val_filelist = [prefix.strip() for prefix in f.readlines()]
    with open(os.path.join(args.data_path,"test.txt")) as f:
        test_filelist = [prefix.strip().split(" ")[0] for prefix in f.readlines()]
    # pretrain分为pretrain和preval
    pretrain_filelist, preval_filelist = np.split(pretrain_filelist, [int(0.99 * len(pretrain_filelist))])
    
    pretrain_filelist = pretrain_filelist.tolist()
    preval_filelist = preval_filelist.tolist()

    pretrain_datalist = []
    preval_datalist = []
    train_datalist = []
    test_datalist = []
    val_datalist = []

    #pretrain和preval分别读取每个单词的开始和截止
    for path in tqdm(pretrain_filelist):
        with open(os.path.join(args.data_path,'pretrain',path+'.mp4'), 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
            video_len = len(vr)

        with open(os.path.join(args.data_path,'pretrain',path + ".txt")) as f:
            txt = [i.strip() for i in f.readlines()]
            sentence = txt[0][7:]
            
            txt = txt[4:]
            id = path.replace('/','-')
            datalist = []
            for line in txt:
                word, start, end, _ = line.split(" ")
                start, end = floor(float(start) * 25), ceil(float(end) * 25) # 不能同时用floor或ceil，会出现start < end的情况
                datalist.append(
                    {"start": start, "end": end, "word": word}
                )
            
        pretrain_datalist.append({
            "id": id,
            'sentence':sentence,
            "words": datalist,
            "path":"pretrain/"+path,
            "video_len": video_len
        })
                
    for path in tqdm(preval_filelist):
        with open(os.path.join(args.data_path,'pretrain',path+'.mp4'), 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
            video_len = len(vr)

        with open(os.path.join(args.data_path,'pretrain',path + ".txt")) as f:
            txt = [i.strip() for i in f.readlines()]
            sentence = txt[0][7:]
            
            txt = txt[4:]
            id = path.replace('/','-')
            datalist = []
            for line in txt:
                word, start, end, _ = line.split(" ")
                start, end = floor(float(start) * 25), ceil(float(end) * 25) # 不能同时用floor或ceil，会出现start < end的情况
                datalist.append(
                    {"start": start, "end": end, "word": word}
                )
            
        preval_datalist.append({
            "id": id,
            'sentence':sentence,
            "words": datalist,
            "path":"pretrain/"+path,
            "video_len": video_len,
        })
            
    #traintestval只用读每个单词即可
    for path in tqdm(train_filelist):
        with open(os.path.join(args.data_path,'main',path+'.mp4'), 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
            video_len = len(vr)

        with open(os.path.join(args.data_path,'main',path + ".txt")) as f:
            txt = [i.strip() for i in f.readlines()]
            sentence = txt[0][7:]
            id = path.replace('/','-')
            
        train_datalist.append({
            "id": id,
            'sentence':sentence,
            "path":"main/"+path,
            "video_len": video_len
        })

    for path in tqdm(test_filelist):
        with open(os.path.join(args.data_path,'main',path+'.mp4'), 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
            video_len = len(vr)

        with open(os.path.join(args.data_path,'main',path + ".txt")) as f:
            txt = [i.strip() for i in f.readlines()]
            sentence = txt[0][7:]
            id = path.replace('/','-')
            
        test_datalist.append({
            "id": id,
            'sentence':sentence,
            "path":"main/"+path,
            "video_len": video_len
        })

    for path in tqdm(test_filelist):
        with open(os.path.join(args.data_path,'main',path+'.mp4'), 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
            video_len = len(vr)

        with open(os.path.join(args.data_path,'main',path + ".txt")) as f:
            txt = [i.strip() for i in f.readlines()]
            sentence = txt[0][7:]
            id = path.replace('/','-')
            
        test_datalist.append({
            "id": id,
            'sentence':sentence,
            "path":"main/"+path, 
            "video_len": video_len
        })
        
    #保存
    np.savez(
        args.save_path,
        pretrain_datalist=pretrain_datalist,
        preval_datalist=preval_datalist,
        train_datalist=train_datalist,
        test_datalist=test_datalist,
        val_datalist=val_datalist,
    )


if __name__ == '__main__':
    args = parse_args()
    mk_datalist(args)