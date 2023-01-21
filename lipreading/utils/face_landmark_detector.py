import numpy as np
import onnxruntime
from numba import njit

import os.path as osp
import numpy as np
# # from utils.io import _load

# # make_abs_path = lambda fn: osp.join(osp.dirname(osp.realpath(__file__)), fn)


# def _to_ctype(arr):
#     if not arr.flags.c_contiguous:
#         return arr.copy(order='C')
#     return arr

# def _load(fp):
#     suffix = _get_suffix(fp)
#     if suffix == 'npy':
#         return np.load(fp)
#     elif suffix == 'pkl':
#         return pickle.load(open(fp, 'rb'))

class BFMModel(object):
    def __init__(self, bfm_fp, shape_dim=40, exp_dim=10):
        bfm = _load(bfm_fp)
        self.u = bfm.get('u').astype(np.float32)  # fix bug
        self.w_shp = bfm.get('w_shp').astype(np.float32)[..., :shape_dim]
        self.w_exp = bfm.get('w_exp').astype(np.float32)[..., :exp_dim]
        if osp.split(bfm_fp)[-1] == 'bfm_noneck_v3.pkl':
            self.tri = _load(make_abs_path('../configs/tri.pkl'))  # this tri/face is re-built for bfm_noneck_v3
        else:
            self.tri = bfm.get('tri')

        self.tri = _to_ctype(self.tri.T).astype(np.int32)
        self.keypoints = bfm.get('keypoints').astype(np.)  # fix bug
        w = np.concatenate((self.w_shp, self.w_exp), axis=1)
        self.w_norm = np.linalg.norm(w, axis=0)

        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]


# class FaceAlignment:
#     """TDDFA_ONNX: the ONNX version of Three-D Dense Face Alignment (TDDFA)"""

#     def __init__(self, bfm_onnx_file, **kvs):
#         # torch.set_grad_enabled(False)

#         # load onnx version of BFM
#         bfm_fp = kvs.get('bfm_fp', make_abs_path('configs/bfm_noneck_v3.pkl'))
#         bfm_onnx_fp = bfm_fp.replace('.pkl', '.onnx')
#         if not osp.exists(bfm_onnx_fp):
#             convert_bfm_to_onnx(
#                 bfm_onnx_fp,
#                 shape_dim=kvs.get('shape_dim', 40),
#                 exp_dim=kvs.get('exp_dim', 10)
#             )
#         self.bfm_session = onnxruntime.InferenceSession(bfm_onnx_file,providers=[('CUDAExecutionProvider', {'device_id': 1,})])

#         # load for optimization
#         bfm = BFMModel(bfm_fp, shape_dim=kvs.get('shape_dim', 40), exp_dim=kvs.get('exp_dim', 10))
#         self.tri = bfm.tri
#         self.u_base, self.w_shp_base, self.w_exp_base = bfm.u_base, bfm.w_shp_base, bfm.w_exp_base

#         # config
#         self.gpu_mode = kvs.get('gpu_mode', False)
#         self.gpu_id = kvs.get('gpu_id', 0)
#         self.size = kvs.get('size', 120)

#         param_mean_std_fp = kvs.get(
#             'param_mean_std_fp', make_abs_path(f'configs/param_mean_std_62d_{self.size}x{self.size}.pkl')
#         )

#         onnx_fp = kvs.get('onnx_fp', kvs.get('checkpoint_fp').replace('.pth', '.onnx'))

#         # convert to onnx online if not existed
#         if onnx_fp is None or not osp.exists(onnx_fp):
#             print(f'{onnx_fp} does not exist, try to convert the `.pth` version to `.onnx` online')
#             onnx_fp = convert_to_onnx(**kvs)

#         self.session = onnxruntime.InferenceSession(onnx_fp, None)

#         # params normalization config
#         r = _load(param_mean_std_fp)
#         self.param_mean = r.get('mean')
#         self.param_std = r.get('std')

#     def __call__(self, img_ori, objs, **kvs):
#         # Crop image, forward to get the param
#         param_lst = []
#         roi_box_lst = []

#         crop_policy = kvs.get('crop_policy', 'box')
#         for obj in objs:
#             if crop_policy == 'box':
#                 # by face box
#                 roi_box = parse_roi_box_from_bbox(obj)
#             elif crop_policy == 'landmark':
#                 # by landmarks
#                 roi_box = parse_roi_box_from_landmark(obj)
#             else:
#                 raise ValueError(f'Unknown crop policy {crop_policy}')

#             roi_box_lst.append(roi_box)
#             img = crop_img(img_ori, roi_box)
#             img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
#             img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
#             img = (img - 127.5) / 128.

#             inp_dct = {'input': img}

#             param = self.session.run(None, inp_dct)[0]
#             param = param.flatten().astype(np.float32)
#             param = param * self.param_std + self.param_mean  # re-scale
#             param_lst.append(param)

#         return param_lst, roi_box_lst

#     def recon_vers(self, param_lst, roi_box_lst, **kvs):
#         dense_flag = kvs.get('dense_flag', False)
#         size = self.size

#         ver_lst = []
#         for param, roi_box in zip(param_lst, roi_box_lst):
#             R, offset, alpha_shp, alpha_exp = _parse_param(param)
#             if dense_flag:
#                 inp_dct = {
#                     'R': R, 'offset': offset, 'alpha_shp': alpha_shp, 'alpha_exp': alpha_exp
#                 }
#                 pts3d = self.bfm_session.run(None, inp_dct)[0]
#                 pts3d = similar_transform(pts3d, roi_box, size)
#             else:
#                 pts3d = R @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp). \
#                     reshape(3, -1, order='F') + offset
#                 pts3d = similar_transform(pts3d, roi_box, size)

#             ver_lst.append(pts3d)

#         return ver_lst
