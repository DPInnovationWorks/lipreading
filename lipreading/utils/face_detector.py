import numpy as np
import onnxruntime
from numba import njit

@njit(cache=True)
def nms(dets, thresh=0.4):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

@njit(fastmath=True, cache=True)
def single_distance2bbox(point, distance, stride):
    """
    Fast conversion of single bbox distances to coordinates
    :param point: Anchor point
    :param distance: Bbox distances from anchor point
    :param stride: Current stride scale
    :return: bbox
    """
    distance[0] = point[0] - distance[0] * stride
    distance[1] = point[1] - distance[1] * stride
    distance[2] = point[0] + distance[2] * stride
    distance[3] = point[1] + distance[3] * stride
    return distance


@njit(fastmath=True, cache=True)
def single_distance2kps(point, distance, stride):
    """
    Fast conversion of single keypoint distances to coordinates
    :param point: Anchor point
    :param distance: Keypoint distances from anchor point
    :param stride: Current stride scale
    :return: keypoint
    """
    for ix in range(0, distance.shape[0], 2):
        distance[ix] = distance[ix] * stride + point[0]
        distance[ix + 1] = distance[ix + 1] * stride + point[1]
    return distance


@njit(fastmath=True, cache=True)
def generate_proposals(score_blob, bbox_blob, kpss_blob, stride, anchors, threshold, score_out, bbox_out, kpss_out,
                       offset):
    """
    Convert distances from anchors to actual coordinates on source image
    and filter proposals by confidence threshold.
    Uses preallocated np.ndarrays for output.
    :param score_blob: Raw scores for stride
    :param bbox_blob: Raw bbox distances for stride
    :param kpss_blob: Raw keypoints distances for stride
    :param stride: Stride scale
    :param anchors: Precomputed anchors for stride
    :param threshold: Confidence threshold
    :param score_out: Output scores np.ndarray
    :param bbox_out: Output bbox np.ndarray
    :param kpss_out: Output key points np.ndarray
    :param offset: Write offset for output arrays
    :return:
    """

    total = offset

    for ix in range(0, anchors.shape[0]):
        if score_blob[ix, 0] > threshold:
            score_out[total] = score_blob[ix]
            bbox_out[total] = single_distance2bbox(anchors[ix], bbox_blob[ix], stride)
            kpss_out[total] = single_distance2kps(anchors[ix], kpss_blob[ix], stride)
            total += 1

    return score_out, bbox_out, kpss_out, total


# @timing
@njit(fastmath=True, cache=True)
def filter(bboxes_list: np.ndarray, kpss_list: np.ndarray,
           scores_list: np.ndarray, nms_threshold: float = 0.4):
    """
    Filter postprocessed network outputs with NMS
    :param bboxes_list: List of bboxes (np.ndarray)
    :param kpss_list: List of keypoints (np.ndarray)
    :param scores_list: List of scores (np.ndarray)
    :return: Face bboxes with scores [t,l,b,r,score], and key points
    """

    pre_det = np.hstack((bboxes_list, scores_list))
    keep = nms(pre_det, thresh=nms_threshold)
    keep = np.asarray(keep)
    det = pre_det[keep, :]
    kpss = kpss_list[keep, :]
    kpss = kpss.reshape((kpss.shape[0], -1, 2))

    return det, kpss


class FaceDetector:

    def __init__(self, onnx_file,provider):
        assert provider in ["Tensorrt","CUDA"]
        
        if provider == 'Tensorrt':
            self.session = onnxruntime.InferenceSession(onnx_file,providers=[('TensorrtExecutionProvider', {'trt_fp16_enable': True})])
        else:
            self.session = onnxruntime.InferenceSession(onnx_file,providers=['CUDAExecutionProvider'])
    
        self.center_cache = {}
        self.nms_threshold = 0.4
        self._anchor_ratio = 1.0
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        # Preallocate reusable arrays for proposals
        self.input_size = (640,640)

        max_prop_len = self._get_max_prop_len(self.input_size,
                                              self._feat_stride_fpn,
                                              self._num_anchors)
        self.input_name = 'input.1'
        self.output_names = ['score_8', 'score_16', 'score_32', 'bbox_8', 'bbox_16', 'bbox_32', 'kps_8', 'kps_16', 'kps_32']
        self.score_list = np.zeros((max_prop_len, 1), dtype='float32')
        self.bbox_list = np.zeros((max_prop_len, 4), dtype='float32')
        self.kpss_list = np.zeros((max_prop_len, 10), dtype='float32')

    @staticmethod
    def _get_max_prop_len(input_shape, feat_strides, num_anchors):
        """
        Estimate maximum possible number of proposals returned by network
        :param input_shape: maximum input shape of model (i.e (1, 3, 640, 640))
        :param feat_strides: model feature strides (i.e. [8, 16, 32])
        :param num_anchors: model number of anchors (i.e 2)
        :return:
        """

        ln = 0
        pixels = input_shape[0] * input_shape[1]
        for e in feat_strides:
            ln += pixels / (e * e) * num_anchors
        return int(ln)
    
    # @timing
    def detect(self, imgs, threshold=0.5):
        """
        Run detection pipeline for provided image
        :param img: Raw image as nd.ndarray with HWC shape
        :param threshold: Confidence threshold
        :return: Face bboxes with scores [t,l,b,r,score], and key points
        """

        batch_size =  imgs.shape[0]
        net_outs = self._forward(imgs)

        dets_list = []
        kpss_list = []

        bboxes_by_img, kpss_by_img, scores_by_img = self._postprocess(net_outs,batch_size, self.input_size[1], self.input_size[0], threshold)

        for e in range(batch_size):
            det, kpss = filter(
                bboxes_by_img[e], kpss_by_img[e], scores_by_img[e], self.nms_threshold)

            dets_list.append(det)
            kpss_list.append(kpss)

        return dets_list, kpss_list



    # @timing
    @staticmethod
    def _build_anchors(input_height, input_width, strides, num_anchors):
        """
        Precompute anchor points for provided image size
        :param input_height: Input image height
        :param input_width: Input image width
        :param strides: Model strides
        :param num_anchors: Model num anchors
        :return: box centers
        """

        centers = []
        for stride in strides:
            height = input_height // stride
            width = input_width // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            centers.append(anchor_centers)
        return centers


    def _forward(self, blob):
        """
        Send input data to inference backend.
        :param blob: Preprocessed image of shape NCHW or None
        :return: network outputs
        """

        net_outs = self.session.run(self.output_names, {self.input_name : blob})
        return net_outs

    # @timing
    def _postprocess(self, net_outs,batch_size, input_height, input_width, threshold):
        """
        Precompute anchor points for provided image size and process network outputs
        :param net_outs: Network outputs
        :param input_height: Input image height
        :param input_width: Input image width
        :param threshold: Confidence threshold
        :return: filtered bboxes, keypoints and scores
        """

        key = (input_height, input_width)

        if not self.center_cache.get(key):
            self.center_cache[key] = self._build_anchors(input_height, input_width, self._feat_stride_fpn,
                                                         self._num_anchors)
        anchor_centers = self.center_cache[key]
        bboxes, kpss, scores = self._process_strides(net_outs,batch_size, threshold, anchor_centers)
        return bboxes, kpss, scores

    def _process_strides(self, net_outs,batch_size, threshold, anchor_centers):
        """
        Process network outputs by strides and return results proposals filtered by threshold
        :param net_outs: Network outputs
        :param threshold: Confidence threshold
        :param anchor_centers: Precomputed anchor centers for all strides
        :return: filtered bboxes, keypoints and scores
        """



        bboxes_by_img = []
        kpss_by_img = []
        scores_by_img = []

        for n_img in range(batch_size):
            offset = 0
            for idx, stride in enumerate(self._feat_stride_fpn):
                score_blob = net_outs[idx][n_img]
                bbox_blob = net_outs[idx + self.fmc][n_img]
                kpss_blob = net_outs[idx + self.fmc * 2][n_img]
                stride_anchors = anchor_centers[idx]
                self.score_list, self.bbox_list, self.kpss_list, total = generate_proposals(score_blob, bbox_blob,
                                                                                            kpss_blob, stride,
                                                                                            stride_anchors, threshold,
                                                                                            self.score_list,
                                                                                            self.bbox_list,
                                                                                            self.kpss_list, offset)
                offset = total

            bboxes_by_img.append(np.copy(self.bbox_list[:offset]))
            kpss_by_img.append(np.copy(self.kpss_list[:offset]))
            scores_by_img.append(np.copy(self.score_list[:offset]))

        return bboxes_by_img, kpss_by_img, scores_by_img