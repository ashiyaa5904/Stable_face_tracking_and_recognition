from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
import cv2
import torch
import numpy as np

class DeepsortFaceTracker:
    def __init__(self):
        pass

    def tracker(self):
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=False)
        return deepsort

    def start_tracking(self,deepsort,xywhs,confs,im0):
        xywhs = torch.Tensor(xywhs)
        confs = torch.Tensor(confs)
        # Pass detections to deepsort
        outputs = deepsort.update(xywhs, confs, im0)
        return outputs

