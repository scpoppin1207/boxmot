# detectron2_infer.py
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

def build_predictor(model_yaml="COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml", score_thresh=0.5, device="cuda"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
    # cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
    cfg.MODEL.WEIGHTS = "/home/scp_recon/thirdparty/detectron2/weights/model_final_a3ec72.pkl"  # 自定义权重路径
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    predictor = DefaultPredictor(cfg)
    return predictor

def detect_and_pack(predictor, bgr_image):
    outputs = predictor(bgr_image)            # detectron2 Results
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else np.zeros((0,4))
    labels = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros((0,), dtype=int)
    scores = instances.scores.numpy() if instances.has("scores") else np.zeros((0,))
    # masks: boolean array (N, H, W) - aligned to original image size
    if instances.has("pred_masks"):
        masks = instances.pred_masks.numpy()  # bool (N, H, W)
        # convert to torchvision style: (N,1,H,W) uint8
        masks_torchvision = (masks.astype("uint8")[:, None, :, :])  # shape (N,1,H,W)
    else:
        masks_torchvision = None

    # Pack into torchvision-like dict
    result = {
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "masks": masks_torchvision
    }
    return result

if __name__ == "__main__":
    predictor = build_predictor(device="cuda")   # or "cpu"
    img = cv2.imread("input.jpg")               # BGR
    res = detect_and_pack(predictor, img)
    print("boxes shape:", res["boxes"].shape)
    print("labels shape:", res["labels"].shape)
    print("scores shape:", res["scores"].shape)
    if res["masks"] is not None:
        print("masks shape:", res["masks"].shape)
