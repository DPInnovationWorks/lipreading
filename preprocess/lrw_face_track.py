import numpy as np
import io
import os
from scenedetect import SceneManager, open_video, ContentDetector
from decord import VideoReader,cpu
import matplotlib.pyplot as plt
import cv2
from lipreading.utils.face_tracker.bot_sort import BoTSORT
from lipreading.utils.face_recognition import FaceRecognizer

