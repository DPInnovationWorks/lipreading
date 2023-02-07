from scenedetect import SceneManager, open_video, ContentDetector
from lipreading.utils.face_tracker.bot_sort import BoTSORT
from decord import VideoReader,cpu
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import math
from tqdm import tqdm

if __name__ == '__main__':
    # contain bbox、kps and reid feature
    root = '/group_homes/public_cluster/home/share/LipReadingGroup/data/LRS2-preprocess/face_detect_results_v2'
    data_root = '/group_homes/public_cluster/home/share/LipReadingGroup/data/LRS2'
    pre_results = os.listdir(root)
    for item in tqdm(pre_results):
        annotation = np.load(os.path.join(root,item),allow_pickle=True).item()
        video_path = os.path.join(data_root,annotation['path']+'.mp4')
        scence_threshold = 27
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(threshold=scence_threshold))
        # Detect all scenes in video from current position to end.
        scene_manager.detect_scenes(video)
        # 根据视频分割读取片段
        seg = scene_manager.get_scene_list()
        with open(video_path, 'rb') as f:
            vr = VideoReader(f, ctx=cpu())
        # 读取检测结果
        all_bboxes,all_reid_feats = annotation['bboxes'],annotation['feats']
        tracks = []
        tracker = BoTSORT(None,0.4,0.1,0.4,0.7,30,0.5,0.25,25)

        if len(seg) > 0:
            for clip in seg[:-1]:
                start,end = clip
                start,end = start.frame_num, end.frame_num
                clip = vr[start:end].asnumpy()
                bboxes,kpss = all_bboxes[start:end],all_reid_feats[start:end]
                for index,(frame,bbox,kps) in enumerate(zip(clip,bboxes,kpss)):
                    online_targets = tracker.update(
                        bbox.astype(np.float32),
                        kps.astype(np.float32),
                        frame,reid,cal_reid_only)
                    tracks.append(online_targets)
                    if index == len(clip) - 2:
                        # 当前clip最后一帧，提取reid特征
                        reid = False
                        cal_reid_only = True
                    elif reid:
                        reid = False
                        cal_reid_only = False

                reid = True
                cal_reid_only = False

            start,_ = seg[-1]
            start = start.frame_num
            clip = vr[start:].asnumpy()
            bboxes,kpss = all_bboxes[start:],all_kpss[start:]
            for frame,bbox,kps in zip(clip,bboxes,kpss):
                online_targets = tracker.update(
                    bbox.astype(np.float32),
                    kps.astype(np.float32),
                    frame,reid,cal_reid_only)
                tracks.append(online_targets)
                if reid:
                    reid = False
                    cal_reid_only = False
        else:
            clip = vr[:].asnumpy()
            bboxes,kpss = all_bboxes,all_kpss
            for frame,bbox,kps in zip(clip,bboxes,kpss):
                online_targets = tracker.update(
                    bbox.astype(np.float32),
                    kps.astype(np.float32),
                    frame)
                tracks.append(online_targets)