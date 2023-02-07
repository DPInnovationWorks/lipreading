import numpy as np
import os
from decord import VideoReader,cpu
from tqdm import tqdm
from lipreading.utils.face_recognition import FaceRecognizer
from lipreading.utils.face_align import norm_crop_batched

if __name__ == '__main__':
    root = '/group_homes/public_cluster/home/share/LipReadingGroup/data/LRS2-preprocess/face_detect_results'
    data_root = '/group_homes/public_cluster/home/share/LipReadingGroup/data/LRS2'
    det_results = os.listdir(root)
    exists = os.listdir('/group_homes/public_cluster/home/share/LipReadingGroup/data/LRS2-preprocess/face_detect_results_v2/')
    reid_model = FaceRecognizer('/group_homes/public_cluster/home/share/LipReadingGroup/lipreading/checkpoints/glintr100.onnx')
    for item in tqdm(det_results):
        if item in exists:
            continue
        annotation = np.load(os.path.join(root,item),allow_pickle=True).item()
        video_path = os.path.join(data_root,annotation['path']+'.mp4')
        with open(video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
        vr = vr[:].asnumpy()
        all_kpss,all_bboxes = annotation['kpss'],annotation['bboxes']
        feats = []
        new_bboxes,new_kpss = [],[]
        try:
            for fid in range(len(vr)):
                frame = vr[fid]

                kpss,bboxes = all_kpss[fid],all_bboxes[fid]
                index = bboxes[:,4] > 0.1
                kpss,bboxes = kpss[index],bboxes[index]

                new_bboxes.append(bboxes)
                new_kpss.append(kpss)

                crops = norm_crop_batched(frame,kpss.astype(np.float32))
                feats.append(np.vstack(reid_model(crop)for crop in crops))
            annotation['feats'] = feats
            annotation['bboxes'] = new_bboxes
            annotation['kpss'] = new_kpss
            np.save('/group_homes/public_cluster/home/share/LipReadingGroup/data/LRS2-preprocess/face_detect_results_v2/'+item,annotation)
        except Exception as e:
            print(e)
            continue
