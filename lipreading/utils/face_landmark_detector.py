import numpy as np
import onnxruntime
import cv2

class FaceLandmarkDetector:
    def __init__(self, onnx_file,mean_face_file,provider) -> None:
        assert provider in ["Tensorrt","CUDA"]
        
        if provider == 'Tensorrt':
            self.session = onnxruntime.InferenceSession(onnx_file,providers=[('TensorrtExecutionProvider', {'trt_fp16_enable': True})])
        else:
            opts = onnxruntime.SessionOptions()
            opts.intra_op_num_threads = 8
            self.session = onnxruntime.InferenceSession(onnx_file,providers=['CUDAExecutionProvider'],sess_options=opts)
        self.size = 120
        self.mean_face = np.load(mean_face_file,allow_pickle=True)
        
    def detect(self,img,bboxes):
        results = []
        for bbox in bboxes:
            crop = self.crop_img(img,bbox)
            crop = cv2.resize(crop, dsize=(self.size,self.size), interpolation=cv2.INTER_LINEAR)
            crop = crop.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
            crop = (crop - 127.5) / 128.
            pts3d = self.session.run(None,{'input': crop})[0]
            pts3d = self.similar_transform(pts3d,bbox)
            results.append(pts3d)
        results = np.stack(results)
        return results
            
    def similar_transform(self,pts3d, bbox):
        pts3d[0, :] -= 1  # for Python compatibility
        pts3d[2, :] -= 1
        pts3d[1, :] = self.size - pts3d[1, :]

        sx, sy, ex, ey = bbox
        scale_x = (ex - sx) / self.size
        scale_y = (ey - sy) / self.size
        pts3d[0, :] = pts3d[0, :] * scale_x + sx
        pts3d[1, :] = pts3d[1, :] * scale_y + sy
        s = (scale_x + scale_y) / 2
        pts3d[2, :] *= s
        pts3d[2, :] -= np.min(pts3d[2, :])
        return np.array(pts3d, dtype=np.float32)
    
    def crop_img(self, img, bbox):
        h, w = img.shape[:2]
        sx, sy, ex, ey = [int(round(_)) for _ in bbox]
        dh, dw = ey - sy, ex - sx
        if len(img.shape) == 3:
            res = np.zeros((dh, dw, 3), dtype=np.uint8)
        else:
            res = np.zeros((dh, dw), dtype=np.uint8)
        if sx < 0:
            sx, dsx = 0, -sx
        else:
            dsx = 0
        if ex > w:
            ex, dex = w, dw - (ex - w)
        else:
            dex = dw
        if sy < 0:
            sy, dsy = 0, -sy
        else:
            dsy = 0
        if ey > h:
            ey, dey = h, dh - (ey - h)
        else:
            dey = dh
        res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
        return res
    
    # def align(self,)