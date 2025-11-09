#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片序列分割跟踪示例
用于处理文件夹中的图片序列，进行实例分割和目标跟踪
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from boxmot import BotSort

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

def build_predictor(model_yaml="COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml", score_thresh=0.5, device="cuda"):
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



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="图片序列分割跟踪")
    parser.add_argument('--source', type=str, required=True, help='图片文件夹路径')
    parser.add_argument('--output', type=str, default='output', help='输出文件夹路径')
    parser.add_argument('--reid-weights', type=str, default='osnet_x0_25_msmt17.pt', help='ReID模型权重路径')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备，cpu或cuda')
    parser.add_argument('--save-vid', action='store_true', help='是否保存为视频')
    parser.add_argument('--vid-fps', type=int, default=15, help='视频帧率')
    parser.add_argument('--save-npy', action='store_true', help='保存跟踪结果为npy文件')
    parser.add_argument('--npy-path', type=str, default='npy_masks', help='npy文件保存路径')
    return parser.parse_args()


def get_color(track_id):
    """为每个跟踪ID生成唯一的颜色"""
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())


def visualize_mask(mask_array):
    """将ID掩码可视化为彩色图像
    
    Args:
        mask_array: 一个2D数组，值为-1(背景)或跟踪ID
    
    Returns:
        colored_mask: 一个彩色的可视化掩码，RGB格式
    """
    # 创建一个全黑的RGB图像
    h, w = mask_array.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 获取所有唯一的ID (除了背景-1)
    unique_ids = np.unique(mask_array)
    unique_ids = unique_ids[unique_ids >= 0]
    
    # 为每个ID上色
    for track_id in unique_ids:
        # 获取此ID的颜色
        color = get_color(int(track_id))
        # 为该ID区域上色
        colored_mask[mask_array == track_id] = color
    
    return colored_mask


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # 创建npy保存目录
    if args.save_npy:
        npy_path = Path(args.npy_path)
        npy_path.mkdir(exist_ok=True, parents=True)
        print(f"将保存跟踪掩码为npy文件到 {npy_path}")
    
    # 设置设备
    device = torch.device(args.device)
    
    # 加载改进版Mask R-CNN模型
    print("正在加载Mask R-CNN...")
    # segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')    
    # segmentation_model = torchvision.models.detection.maskrcnn_mobilenet_v3_large_fpn(weights='DEFAULT')
    # segmentation_model.eval().to(device)

    segmentation_model = build_predictor(device="cuda")
    
    # 初始化跟踪器
    print(f"初始化BotSort跟踪器，使用ReID权重: {args.reid_weights}")
    tracker = BotSort(
        reid_weights=Path(args.reid_weights),
        device=device,
        half=False,
    )
    
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(args.source).glob(f'*{ext}')))
        image_files.extend(list(Path(args.source).glob(f'*{ext.upper()}')))
    image_files = sorted(image_files)
    base_names = [f.name.split('.')[0] for f in image_files]
    if not image_files:
        print(f"在 {args.source} 中未找到图片文件")
        return
    # 读取第一张图片获取图像尺寸
    first_img = cv2.imread(str(image_files[0]))
    img_height, img_width = first_img.shape[:2]
    
    # 视频写入器
    video_writer = None
    if args.save_vid:
        video_path = output_path / 'tracking_result.mp4'
        video_writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            args.vid_fps,
            (img_width, img_height)
        )
    

    # 存储每个ID外观特征
    appearance_instant = {}
    appearance_ema = {}
    frame_record = {}
    area_record = {}
    cls_record = {}
    
    

    track_class = [0,1,2,3,4,5,6,7]  # 只跟踪这些类别，COCO类别ID
    # 处理每一张图片
    for idx, img_path in enumerate(tqdm(image_files, desc="处理图片")):
        # 读取图片
        im = cv2.imread(str(img_path))
        if im is None:
            print(f"无法读取图片 {img_path}")
            continue
        
        # # 将图片转换为tensor并移至设备
        # frame_tensor = torchvision.transforms.functional.to_tensor(im).unsqueeze(0).to(device)
        
        # 运行Mask R-CNN模型检测边界框和掩码
        with torch.no_grad():
            results = detect_and_pack(segmentation_model, im)
            # 'box': (N,4) 每个目标的BB坐标
            # 'labels': (N,) 每个目标的类别标签
            # 'scores': (N,) 每个目标的置信度分数
            # 'masks': (N,1,H,W) 每个目标的分割掩码
        
        # 提取分割结果
        dets = []
        masks = [] 
        confidence_threshold = args.conf_thres
        
        for i, score in enumerate(results['scores']):
            if score >= confidence_threshold:
                # 提取边界框和分数
                x1, y1, x2, y2 = results['boxes'][i]
                conf = score
                cls = results['labels'][i]
                if cls not in track_class:
                    continue
                dets.append([x1, y1, x2, y2, conf, cls])
                
                # 提取掩码并添加到列表
                mask = results['masks'][i,0]# 使用第一个通道（二值掩码）
                masks.append(mask)
        
        # 将检测结果转换为numpy数组 (N x (x, y, x, y, conf, cls))
        if dets:
            dets = np.array(dets)
        else:
            dets = np.empty((0, 6))
        
        # 更新跟踪器
        tracks = tracker.update(dets, im) 
        for t in tracker.active_tracks:
            if t.is_activated:
                track_id = t.id
                # 现在可以安全地访问特征
                curr_feat = t.curr_feat if hasattr(t, 'curr_feat') else None
                smooth_feat = t.smooth_feat if hasattr(t, 'smooth_feat') else None
                conf = t.conf if hasattr(t, 'conf') else None
                if track_id not in appearance_instant:
                    appearance_instant[track_id] = []
                if track_id not in appearance_ema:
                    appearance_ema[track_id] = []
                if track_id not in area_record:
                    area_record[track_id] = []
                if track_id not in cls_record:
                    cls_record[track_id] = []

                appearance_instant[track_id].append(curr_feat)
                appearance_ema[track_id].append(smooth_feat)
                area_record[track_id].append(abs((t.xyxy[2]-t.xyxy[0])*(t.xyxy[3]-t.xyxy[1])))  # x2-x1 * y
                cls_record[track_id].append(t.cls)  # 类别
               

                if track_id not in frame_record:
                    frame_record[track_id] = [t.start_frame]
                else:
                    if len(frame_record[track_id])==1:
                        frame_record[track_id].append(t.frame_id)
                    else:
                        frame_record[track_id][1] = t.frame_id


                
        

        # id为某个对象的唯一标识符，通常是跟踪器分配的
        # ind是长度为最大ID数的数组，表示每个ID在dets(也即mask）对应的索引
        
        if args.save_npy:
            # 创建与原图相同高度和宽度的数组，初始化为-1
            track_mask = np.ones((im.shape[0], im.shape[1]), dtype=np.int32) * -1
       
        # 在单个循环中绘制分割掩码和边界框
        if len(tracks) > 0:
            inds = tracks[:, 7].astype('int')  # 获取跟踪索引为整数
            # 使用索引匹配跟踪和掩码
            if len(masks) > 0:
                # 确保索引在有效范围内
                valid_masks = []
                for i in inds:
                    if i < len(masks):
                        # 表示该对象在该帧有有效掩码
                        valid_masks.append(masks[i])
                    else:
                        # 该对象在该帧没有有效掩码
                        # 添加一个空掩码作为占位符
                        valid_masks.append(None)
                masks = valid_masks # 此时共有 len(tracks) 个掩码
            
            # 遍历跟踪和相应的掩码一起绘制它们
            for track_idx, (track, mask) in enumerate(zip(tracks, masks)):
                track_id = int(track[4])  # 提取跟踪ID
                color = get_color(track_id)  # 为每个跟踪使用唯一的颜色
                
                # 绘制分割掩码
                if mask is not None:
                    # 二值化掩码，使用较低的阈值确保捕获更多细节
                    binary_mask = (mask > 0.5).astype(np.uint8)
                    # 检查二值化后的掩码中有多少像素被设置为1
                    active_pixels = np.sum(binary_mask)
                    
                    # 如果需要保存npy，更新跟踪掩码
                    if args.save_npy and active_pixels > 0:
                        # 将当前对象的ID填入掩码对应区域
                        track_mask[binary_mask == 1] = track_id
                        
                    # 将掩码颜色与图像混合
                    im[binary_mask == 1] = im[binary_mask == 1] * 0.5 + np.array(color) * 0.5
                
                # 绘制边界框
                x1, y1, x2, y2 = track[:4].astype('int')
                cv2.rectangle(im, (x1, y1), (x2, y2), color, 2)
                
                # 添加ID、置信度和类别的文本
                conf = track[5]
                cls = track[6]
                cv2.putText(im, f'ID: {track_id}, Conf: {conf:.2f}, Class: {cls}', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存跟踪掩码为npy文件
        if args.save_npy:
            npy_file_path = npy_path / f"{base_names[idx]}.npy"
            np.save(str(npy_file_path), track_mask)
            
            # 可视化跟踪掩码并保存为图像
            vis_path = npy_path / "visualized"
            vis_path.mkdir(exist_ok=True, parents=True)
            
            # 生成彩色可视化
            colored_mask = visualize_mask(track_mask)
            
            # 保存可视化结果
            vis_file_path = vis_path / f"mask_vis_{idx:04d}.jpg"
            cv2.imwrite(str(vis_file_path), colored_mask)
            
            # 检查掩码中非背景像素的数量
            non_bg_pixels = np.sum(track_mask >= 0)
            print(f"帧 {idx}: 掩码中有 {non_bg_pixels} 个非背景像素")
            
            # 如果全是背景，输出警告
            if non_bg_pixels == 0:
                print(f"警告: 帧 {idx} 的掩码全是背景 (-1)")
        
        # 保存处理后的图像
        output_img_path = output_path / f"frame_{idx:04d}.jpg"
        cv2.imwrite(str(output_img_path), im)
        
        # 添加到视频
        if args.save_vid and video_writer is not None:
            video_writer.write(im)
    
    # 清理
    if video_writer is not None:
        video_writer.release()
    
    cv2.destroyAllWindows()

    assert len(appearance_instant) == len(appearance_ema) == len(area_record), "instant, ema 长度不一致"
    max_id = max(appearance_instant.keys())
    final_feature_dict = {
        id: {
            "instant": np.array(appearance_instant[id]).mean(axis=0) if len(appearance_instant[id]) > 0 else None,
            "ema": np.array(appearance_ema[id]).mean(axis=0) if len(appearance_ema[id]) > 0 else None,
            "area": np.array(area_record[id]).mean() if len(area_record[id]) > 0 else None,
            "cls": max(set(cls_record[id]), key=cls_record[id].count) if len(cls_record[id]) > 0 else None
        } for id in appearance_instant.keys()
    }

    feat_dim = 1280
    instant_feat_np = np.zeros((max_id, feat_dim), dtype=np.float32)
    ema_feat_np = np.zeros((max_id, feat_dim), dtype=np.float32)
    frame_record_np = np.zeros((max_id, 4), dtype=np.int32)  # start_frame, end_frame, avg_area, cls
    frame_record_np[:,-1] = -1  # cls初始化为-1，表示无效



    for id in final_feature_dict.keys():
        if final_feature_dict[id]['instant'] is not None:
            instant_feat_np[id-1] = final_feature_dict[id]['instant']
        if final_feature_dict[id]['ema'] is not None:
            ema_feat_np[id-1] = final_feature_dict[id]['ema']
        if id in frame_record:
            frame_record_np[id-1][:2] = np.array(frame_record[id], dtype=np.int32)
        if final_feature_dict[id]['area'] is not None:
            frame_record_np[id-1][2] = final_feature_dict[id]['area']  
        if id in cls_record:
            frame_record_np[id-1][3] = final_feature_dict[id]['cls']
        
        

    

    
    np.save(output_path / "instant_feat.npy", instant_feat_np)
    np.save(output_path / "ema_feat.npy", ema_feat_np)
    np.save(output_path / "frame_record.npy", frame_record_np)
    
    # print(f"处理完成。彩色结果保存在 {output_path}, npy掩码保存在 {npy_path}。")




if __name__ == "__main__":
    main()
