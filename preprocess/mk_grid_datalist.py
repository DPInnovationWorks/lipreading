import os
import numpy as np
import lmdb

if __name__ == '__main__':
    total_size = 0
    new_cnt = 0
    for root,dirs,files in os.walk('data/GRID/lip'):
        if len(files) > 0:
            for f in files:
                new_cnt += 1
                total_size += os.path.getsize(os.path.join(root,f))
    
    env = lmdb.open('data/GRID-preprocess/lmdb',map_size=total_size + total_size // 3)
    txn = env.begin(write=True)
    
    cnt = 0
    for root,dirs,files in os.walk('data/GRID/lip'):
        if len(files) > 0:
            for f in files:
                id = root.split('/')[-1]+'-'+f
                with open(os.path.join(root,f),'rb') as f:
                    txn.put(id.encode(),f.read())
                cnt += 1
                if cnt % 1000 == 0:
                    txn.commit()
                    txn = env.begin(write=True)
    txn.commit()

    video_len = {}
    for root,dirs,files in os.walk('data/GRID/lip'):
        if len(files) > 0:
            for f in files:
                if root.split('/')[-1] in video_len:
                    video_len[root.split('/')[-1]] += 1
                else:
                    video_len[root.split('/')[-1]] = 1
    
    datalist = []
    for root,dirs,files in os.walk('data/GRID/GRID_align_txt'):
        for file in files:
            with open(os.path.join(root,file)) as f:
                lines = [line.strip().split(' ') for line in f.readlines()]
                txt = [line[2] for line in lines]
                txt = list(filter(lambda s: not s.upper() in ['SIL'], txt))
                txt = ' '.join(txt).upper()
                datalist.append({'file_name':file.split('.')[0],'sentence':txt,'video_len':video_len[file.split('.')[0]]})
    np.save('data/GRID-preprocess/datalist.npy',datalist)