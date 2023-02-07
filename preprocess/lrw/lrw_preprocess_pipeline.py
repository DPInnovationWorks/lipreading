import numpy as np
from tqdm import tqdm
from lipreading.utils.face_detector import FaceDetector
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from decord import VideoReader,cpu
import os
import cv2

class LRWInferenceDataset(Dataset):
    def __init__(self,datalist_path,data_prefix,previous_result_path,input_size = (640,640)):
        self.datalist = np.load(datalist_path,allow_pickle=True)
        self.input_size = input_size
        self.data_prefix = data_prefix
        self.previous_results = os.listdir(previous_result_path)
        self.datalist = [i for i in self.datalist if i['path'].replace('/','-') not in self.previous_results]
        self.model_ratio = float(self.input_size[1]) / self.input_size[0]
        
    def __getitem__(self, index):
        item = self.datalist[index]
        vidname = os.path.join(self.data_prefix,item['path']+'.mp4')
        with open(vidname, 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
        img_shape = vr[0].asnumpy().shape
        im_ratio = float(img_shape[0]) / img_shape[1]
        
        if im_ratio > self.model_ratio:
            new_height = self.input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = self.input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img_shape[0]
        
        video = []
        for i in range(len(vr)):
            frame = cv2.resize(vr[i].asnumpy(), (new_width, new_height))
            det_img = np.zeros( (self.input_size[1], self.input_size[0], 3), dtype=np.uint8 )
            det_img[:new_height, :new_width, :] = frame
            video.append(det_img)
        video = np.stack(video)
        blob = cv2.dnn.blobFromImages(video, 1.0/128, self.input_size, (127.5, 127.5, 127.5), swapRB=False)
        return blob,det_scale,item['path']
        
    def __len__(self):
        return len(self.datalist)
    
def main():    

    dataset = LRWInferenceDataset('data/LRW-preprocess/datalist.npy','data/LRW','data/LRW-preprocess/face_detect_results')
 
    def custom_collate(batch):
        videos = []
        names = []
        det_scales = []
        length = []
        for video,det_scale,name in batch:
            length.append(video.shape[0])
            videos.append(video)
            names.append(name)
            det_scales.append(det_scale)
        videos = np.vstack(videos)
        return videos,names,det_scales,length
    
    batch_size = 2
    max_seq_len = 5*30
    
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=custom_collate,num_workers=10,pin_memory=True)
    detector = FaceDetector('weight/scrfd_10g_gnkps.onnx','CUDA')
    results = []
    for batch,names,det_scales,length in tqdm(loader):
        bboxes,kpss = detector.detect(batch,0.05)
        bboxes = [i.astype(np.float16) for i in bboxes]
        kpss = [i.astype(np.float16) for i in kpss]
        # TODO: 继续细分batch
        
        start_index = 0
        for i in range(batch_size):
            name = names[i]
            det_scale = det_scales[i]
            batch_len = length[i]
            
            cur_bboxes = [i/det_scale for i in  bboxes[start_index:start_index+batch_len]]
            cur_kpss = [i/det_scale for i in kpss[start_index:start_index+batch_len]] 
            result = {'path':name,'bboxes':cur_bboxes,'kpss':cur_kpss}
            np.save('data/LRW-preprocess/face_detect_results/'+name.replace('/','-')+'.npy',result)

    print('done')



if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES="0" python preprocess/face_detect.py  image_path
# python tools/video_infer.py /root/vedadet/configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py/root/vedadet/configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py ABOUT_00026.mp4