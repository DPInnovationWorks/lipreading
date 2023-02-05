import onnxruntime
import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self, onnx_file):
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 8
        self.rec_model = onnxruntime.InferenceSession(onnx_file,providers=['CUDAExecutionProvider'],sess_options=opts)
        self.input_size = (112,112)
        self.input_mean = 127.5
        self.input_std = 127.5

    def __call__(self, frame):
        frame = np.expand_dims(frame,axis=0)
        blob = cv2.dnn.blobFromImages(frame, 1.0 / self.input_std, self.input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_out = self.rec_model.run(None, {'input.1': blob})
        return net_out[0]