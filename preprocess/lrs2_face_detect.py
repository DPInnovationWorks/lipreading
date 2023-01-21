import numpy as np
from tqdm import tqdm
from lipreading.utils.face_detector import FaceDetector
from tqdm import tqdm
import numpy as np
from decord import VideoReader,cpu
import os
import cv2

def main():    
    
    datalist = np.load('data/LRS2-preprocess/datalist.npz',allow_pickle=True)
    new_datalist = []
    new_datalist.extend(datalist['pretrain_datalist'].tolist())
    new_datalist.extend(datalist['preval_datalist'].tolist())
    new_datalist.extend(datalist['train_datalist'].tolist())
    new_datalist.extend(datalist['val_datalist'].tolist())
    new_datalist.extend(datalist['test_datalist'].tolist())
    datalist = new_datalist
    
    input_size = (640,640)
    model_ratio = float(input_size[1]) / input_size[0]
    detector = FaceDetector('checkpoints/scrfd_10g_gnkps.onnx','CUDA')
    
    for item in tqdm(datalist):
        if os.path.exists('data/LRS2-preprocess/face_detect_results/'+item['path'].replace('/','-')+'.npy'):
            continue
        vidname = os.path.join('data/LRS2',item['path']+'.mp4')
        with open(vidname, 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
        img_shape = vr[0].asnumpy().shape
        im_ratio = float(img_shape[0]) / img_shape[1]
        
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img_shape[0]
        
        video = []
        for i in range(len(vr)):
            frame = cv2.resize(vr[i].asnumpy(), (new_width, new_height))
            det_img = np.zeros( (input_size[1], input_size[0], 3), dtype=np.uint8 )
            det_img[:new_height, :new_width, :] = frame
            video.append(det_img)
        video = np.stack(video)
        blob = cv2.dnn.blobFromImages(video, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=False)
        
        bboxes,kpss = [],[]
        for i in range(0, len(blob), 92):
            chunk = blob[i:i+92]
            
            chunk_bboxes,chunk_kpss = detector.detect(chunk,0.05)
            bboxes.extend([i.astype(np.float16) for i in chunk_bboxes])
            kpss.extend([i.astype(np.float16) for i in chunk_kpss])
                
        cur_bboxes = [np.hstack([i[:,:4]/det_scale,i[:,4:]]) for i in bboxes]
        cur_kpss = [i/det_scale for i in kpss]
        assert len(cur_bboxes) == len(video)
        result = {'path':item['path'],'bboxes':cur_bboxes,'kpss':cur_kpss}
        
        np.save('data/LRS2-preprocess/face_detect_results/'+item['path'].replace('/','-')+'.npy',result)

    print('done')



if __name__ == '__main__':
    main()