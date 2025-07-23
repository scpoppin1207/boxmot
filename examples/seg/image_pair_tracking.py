#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图片对分割跟踪示例
用于处理两帧图片对，进行实例分割和目标跟踪
"""

import argparse
import os
from pathlib import Path
import re
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from boxmot import BotSort
import cv2
import torch
import sys

# 添加 Depth-Anything-V2 模块的上级目录到 sys.path
depth_module_path = '/home/scp_recon/thirdparty/Depth-Anything-V2'
if depth_module_path not in sys.path:
    sys.path.insert(0, depth_module_path)
from depth_anything_v2.dpt import DepthAnythingV2


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="图片对分割跟踪")
    parser.add_argument('--source', type=str, required=True, help='图片文件夹路径')
    parser.add_argument('--all_seq_mask_path', type=str, required=True, help='GT掩码文件夹路径')
    parser.add_argument('--calib_path', type=str,  required=True, help='相机内参文件路径')
    parser.add_argument('--output', type=str, default='output_pairs', help='输出文件夹路径')
    parser.add_argument('--reid-weights', type=str, default='osnet_x0_25_msmt17.pt', help='ReID模型权重路径')
    parser.add_argument('--conf-thres', type=float, default=0.9, help='置信度阈值')
    parser.add_argument('--min-bbox-area', type=int, default=500, help='最小边界框面积阈值（像素）')
    parser.add_argument('--device', type=str, default='cpu', help='运行设备，cpu或cuda')
    parser.add_argument('--save-npy', action='store_true', help='保存跟踪结果为npy文件')
    return parser.parse_args()


def get_color(track_id):
    """为每个跟踪ID生成唯一的颜色"""
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())


def find_image_pairs(source_path):
    """查找所有的图片对
    
    Args:
        source_path: 图片文件夹路径
    
    Returns:
        pairs: [(gt_path, wrap_path, frame_id, cam_id), ...]
    """
    source_path = Path(source_path)
    pairs = []
    
    # 查找所有gt图片
    print(f"在 {source_path} 中查找GT图片...")
    gt_pattern = re.compile(r'gt_(\d+)_cam(\d+)\.png')
    gt_files = list(source_path.glob('gt_*_cam*.png'))
    print(f"找到 {len(gt_files)} 个GT图片")
    
    for gt_file in gt_files:
        match = gt_pattern.match(gt_file.name)
        if match:
            frame_id, cam_id = match.groups()
            
            # 查找对应的wrap图片
            wrap_file = source_path / f'wrap_{frame_id}_cam{cam_id}.png'
            if wrap_file.exists():
                pairs.append((gt_file, wrap_file, frame_id, cam_id))
    
    return sorted(pairs, key=lambda x: (int(x[2]), int(x[3])))  # 按frame_id和cam_id排序


def process_image(image_path, segmentation_model, device, conf_thres):
    """处理单张图片进行目标检测和分割
    
    Returns:
        dets: 检测结果数组
        masks: 掩码列表
        im: 原始图像
    """
    # 读取图片
    im = cv2.imread(str(image_path))
    if im is None:
        print(f"无法读取图片 {image_path}")
        return None, None, None
    
    # 将图片转换为tensor并移至设备
    frame_tensor = torchvision.transforms.functional.to_tensor(im).unsqueeze(0).to(device)
    
    # 运行Mask R-CNN模型检测边界框和掩码
    with torch.no_grad():
        results = segmentation_model(frame_tensor)[0]
    
    # 提取分割结果
    dets = []
    masks = [] 
    track_class = [1, 2, 3, 4, 6, 7, 8]
    
    for i, score in enumerate(results['scores']):
        if score >= conf_thres:
            # 提取边界框和分数
            x1, y1, x2, y2 = results['boxes'][i].cpu().numpy()
            conf = score.item()
            cls = results['labels'][i].item()  
            if cls not in track_class:
                continue
            dets.append([x1, y1, x2, y2, conf, cls])
            
            # 提取掩码并添加到列表
            mask = results['masks'][i, 0].cpu().numpy()  # 使用第一个通道（二值掩码）
            masks.append(mask)
    
    # 将检测结果转换为numpy数组
    if dets:
        dets = np.array(dets)
    else:
        dets = np.empty((0, 6))
    
    return dets, masks, im


def create_tracking_mask(tracks, masks, image_shape):
    """创建跟踪掩码
    
    Returns:
        track_mask: 跟踪掩码数组
        track_info: {track_id: area} 字典
    """
    track_mask = np.ones(image_shape[:2], dtype=np.int32) * -1
    track_info = {}
    
    if len(tracks) > 0:
        inds = tracks[:, 7].astype('int')  # 获取跟踪索引
        
        # 确保索引在有效范围内
        valid_masks = []
        for i in inds:
            if i < len(masks):
                valid_masks.append(masks[i])
            else:
                valid_masks.append(None)
        
        # 遍历跟踪和相应的掩码
        for track, mask in zip(tracks, valid_masks):
            track_id = int(track[4])  # 提取跟踪ID
            
            if mask is not None:
                # 二值化掩码
                binary_mask = (mask > 0.5).astype(np.uint8)
                active_pixels = np.sum(binary_mask)
                
                if active_pixels > 0:
                    # 将当前对象的ID填入掩码对应区域
                    track_mask[binary_mask == 1] = track_id
                    track_info[track_id] = active_pixels
    
    return track_mask, track_info


def calculate_bbox_iou(bbox1, bbox2):
    """计算两个边界框的IoU
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 格式的边界框
    
    Returns:
        iou: 边界框IoU值
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 计算交集区域
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    
    # 检查是否有交集
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    # 计算交集面积
    intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 计算两个边界框的面积
    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # 计算并集面积
    union_area = bbox1_area + bbox2_area - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    
    return iou


def calculate_motion_metrics(bbox1, bbox2):
    """计算更敏感的运动指标
    
    Args:
        bbox1, bbox2: [x1, y1, x2, y2] 格式的边界框
    
    Returns:
        dict: 包含多种运动指标的字典
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 去除突变情况
    # Boundbox突变，不进行动态积分计算
    horizon_change_ratio = abs((x2_2 - x1_2) - (x2_1 - x1_1))/abs(x2_1 - x1_1)
    vertical_change_ratio = abs((y2_2 - y1_2) - (y2_1 - y1_1))/abs(y2_1 - y1_1)
    if horizon_change_ratio > 0.3 or vertical_change_ratio > 0.3:
        return None
    

    
    # 1. 中心点位移距离
    center1 = np.array([(x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2])
    center2 = np.array([(x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2])
    center_displacement = np.linalg.norm(center2 - center1)

    
    # 2. 边界框面积变化率
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    area_change_ratio = abs(area2 - area1) / area1 if area1 > 0 else 0
    
    # 3. 宽高比变化
    aspect_ratio1 = (x2_1 - x1_1) / (y2_1 - y1_1) if (y2_1 - y1_1) > 0 else 0
    aspect_ratio2 = (x2_2 - x1_2) / (y2_2 - y1_2) if (y2_2 - y1_2) > 0 else 0
    aspect_ratio_change = abs(aspect_ratio2 - aspect_ratio1)
    if aspect_ratio_change > 0.15:
        return None
    
    
    # 4. 各边位移的标准差（形状变化指标）
    edges1 = np.array([x1_1, y1_1, x2_1, y2_1])
    edges2 = np.array([x1_2, y1_2, x2_2, y2_2])
    edge_displacements = np.abs(edges2 - edges1)
    shape_variance = np.std(edge_displacements)
    
    # 5. 运动强度分数（综合指标）
    # 归一化各个指标并加权组合
    bbox_size = np.sqrt(area1)  # 用于归一化位移
    normalized_displacement = center_displacement / bbox_size if bbox_size > 0 else 0

    
    # 运动强度分数：综合中心位移、面积变化和形状变化
    motion_intensity = (
        0.8 * normalized_displacement +  # 中心点位移
        0.1 * area_change_ratio +        # 面积变化
        0.1 * shape_variance / bbox_size if bbox_size > 0 else 0  # 形状变化
    )
    
    return motion_intensity


def filter_small_bboxes(tracks, min_area):
    """过滤掉面积小于阈值的边界框
    
    Args:
        tracks: 跟踪结果数组
        min_area: 最小面积阈值
    
    Returns:
        filtered_tracks: 过滤后的跟踪结果
    """
    if len(tracks) == 0:
        return tracks
    
    filtered_tracks = []
    for track in tracks:
        x1, y1, x2, y2 = track[:4]
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area >= min_area:
            filtered_tracks.append(track)
    
    return np.array(filtered_tracks) if filtered_tracks else np.empty((0, tracks.shape[1]))


def filter_by_class_average_area(tracks, threshold_ratio=0.5):
    """根据每个类别的平均面积过滤跟踪结果
    
    Args:
        tracks: 跟踪结果数组，每行格式为 [x1, y1, x2, y2, track_id, conf, cls, det_ind]
        threshold_ratio: 面积阈值比例，默认0.5（即平均面积的50%）
    
    Returns:
        filtered_tracks: 过滤后的跟踪结果
    """
    if len(tracks) == 0:
        return tracks
    
    # 计算每个类别的平均面积
    class_areas = {}
    class_counts = {}
    
    for track in tracks:
        x1, y1, x2, y2 = track[:4]
        cls = int(track[6])  # 类别在第6列
        bbox_area = (x2 - x1) * (y2 - y1)
        
        if cls not in class_areas:
            class_areas[cls] = 0
            class_counts[cls] = 0
        
        class_areas[cls] += bbox_area
        class_counts[cls] += 1
    
    # 计算每个类别的平均面积
    class_avg_areas = {}
    for cls in class_areas:
        class_avg_areas[cls] = class_areas[cls] / class_counts[cls]
        print(f"  类别 {cls}: 平均面积 {class_avg_areas[cls]:.1f}，阈值 {class_avg_areas[cls] * threshold_ratio:.1f}")
    
    # 过滤跟踪结果
    filtered_tracks = []
    filtered_count = 0
    
    for track in tracks:
        x1, y1, x2, y2 = track[:4]
        cls = int(track[6])
        bbox_area = (x2 - x1) * (y2 - y1)
        
        threshold_area = class_avg_areas[cls] * threshold_ratio
        
        if bbox_area >= threshold_area:
            filtered_tracks.append(track)
        else:
            filtered_count += 1
    
    if filtered_count > 0:
        print(f"  基于类别平均面积过滤掉 {filtered_count} 个跟踪结果")
    
    return np.array(filtered_tracks) if filtered_tracks else np.empty((0, tracks.shape[1]))


def get_mask_by_track_id(tracks, masks, track_id):
    """根据跟踪ID获取对应的掩码
    
    Args:
        tracks: 跟踪结果数组
        masks: 掩码列表
        track_id: 跟踪ID
    
    Returns:
        mask: 对应的掩码，如果未找到返回None
    """
    for i, track in enumerate(tracks):
        if int(track[4]) == track_id:
            # 获取跟踪索引
            mask_idx = int(track[7])
            if mask_idx < len(masks):
                return masks[mask_idx]
    return None


def get_bbox_by_track_id(tracks, track_id):
    """根据跟踪ID获取边界框
    
    Args:
        tracks: 跟踪结果数组
        track_id: 跟踪ID
    
    Returns:
        bbox: [x1, y1, x2, y2] 格式的边界框，如果未找到返回None
    """
    for track in tracks:
        if int(track[4]) == track_id:
            return track[:4].astype('float')
    return None


def visualize_tracking_result(image, tracks, masks, id_mapping, rgb_diff_vis):
    """可视化跟踪结果"""
    vis_image = image.copy()
    rgb_diff_vis = rgb_diff_vis.copy()
    
    if len(tracks) > 0:
        inds = tracks[:, 7].astype('int')
        
        # 确保索引在有效范围内
        valid_masks = []
        for i in inds:
            if i < len(masks):
                valid_masks.append(masks[i])
            else:
                valid_masks.append(None)
        
        # 绘制掩码和边界框
        for track, mask in zip(tracks, valid_masks):
            track_id = int(track[4])
            if track_id not in id_mapping.keys():
                print(f"跳过未映射的ID: {track_id}")
                continue
            else:
                color = get_color(id_mapping[track_id]) 
            
            # 绘制分割掩码
            if mask is not None:
                binary_mask = (mask > 0.5).astype(np.uint8)
                if np.sum(binary_mask) > 0:
                    # 将掩码颜色与图像混合
                    vis_image[binary_mask == 1] = vis_image[binary_mask == 1] * 0.5 + np.array(color) * 0.5
                    rgb_diff_vis[binary_mask == 1] = rgb_diff_vis[binary_mask == 1] * 0.5 + np.array(color) * 0.5
            
            # 绘制边界框
            x1, y1, x2, y2 = track[:4].astype('int')
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(rgb_diff_vis, (x1, y1), (x2, y2), color, 2)
            
            # 添加ID标签
            cv2.putText(vis_image, f'ID: {id_mapping[track_id]}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(rgb_diff_vis, f'ID: {id_mapping[track_id]}', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_image, rgb_diff_vis


def create_id_mapping(mask_A, mask_B, min_iou=0.5):
    """
    创建 mask_B ID 到 mask_A ID 的映射字典，基于 IoU
    
    Args:
        mask_A: 参考掩码数组 (H, W)，值为 -1(背景) 或跟踪ID
        mask_B: 待映射掩码数组 (H, W)，值为 -1(背景) 或跟踪ID  
        min_iou: 最小IoU阈值，用于确定是否为同一实例
    
    Returns:
        mapping_dict: {mask_B_id: mask_A_id} 的映射字典
    """
    # 获取所有非背景的唯一ID
    ids_A = np.unique(mask_A[mask_A >= 0])
    ids_B = np.unique(mask_B[mask_B >= 0])
    
    mapping_dict = {}
    used_A_ids = set()
    
    for id_B in ids_B:
        mask_B_region = (mask_B == id_B)
        best_match_id = -1
        best_iou = 0
        
        for id_A in ids_A:
            if id_A in used_A_ids:
                continue
                
            mask_A_region = (mask_A == id_A)
            
            # 计算IoU
            intersection = np.sum(mask_A_region & mask_B_region)
            union = np.sum(mask_A_region | mask_B_region)
            iou = intersection / union if union > 0 else 0
            
            if iou > best_iou and iou >= min_iou:
                best_iou = iou
                best_match_id = id_A
        
        if best_match_id != -1:
            mapping_dict[id_B] = best_match_id
            used_A_ids.add(best_match_id)
            # print(f"映射: mask_B ID {id_B} -> mask_A ID {best_match_id} (IoU: {best_iou:.3f})")
    
    return mapping_dict

def load_intrinsic_matrix(all_calib_path):
    Ks = [] #cam0 的内参矩阵列表
    print(f"加载相机内参文件: {all_calib_path}")
    calib_paths = os.listdir(all_calib_path)
    calib_paths = [c for c in calib_paths if c.endswith('.txt')]
    calib_paths = sorted(calib_paths, key=lambda x: int(x.split('.')[0]))  # 按照文件名中的数字排序
    for calib_file in calib_paths:
        with open(os.path.join(all_calib_path, calib_file)) as f:
            calib_data = f.readlines()
            L = [list(map(float, line.split()[1:])) for line in calib_data]
            K_all_cam = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3]  # 相机内参旋转矩阵
            Ks.append(K_all_cam[0])
    Ks = np.array(Ks)  # 转换为numpy数组 # (N, 3, 3)
    return Ks



    # with open(os.path.join(args.source_path, 'calib', car_id + '.txt')) as f:
    #             calib_data = f.readlines()
    #             L = [list(map(float, line.split()[1:])) for line in calib_data] # 长度为行数，每行12个float元素

    #         Ks = np.array(L[:5]).reshape(-1, 3, 4)[:, :, :3] # 相机内参旋转矩阵

def main():
    """主函数"""
    args = parse_args()
    load_intrinsic_matrix(args.calib_path)
    
    # 创建输出目录
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    print(f"输出目录: {output_path}")
    
    # 设置设备（添加CUDA可用性检查）
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，回退到CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    # depth_model = DepthAnythingV2(**model_configs[encoder])
    # depth_model.load_state_dict(torch.load(f'/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    # depth_model = depth_model.to(device).eval()
    # print(f"成功加载Depth Anything V2模型，编码器: {encoder}")

    # raw_img = cv2.imread('your/image/path')
    # depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    
    # 加载Mask R-CNN模型
    segmentation_model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')    
    segmentation_model.eval().to(device)
    print("成功加载Mask R-CNN模型")
    
    # 查找所有图片对
    seq_ID_score = {} # 存储ID映射后的动态得分
    seq_ID_score_count = {} # 存储ID映射后的动态得分计数次数
    mapping_list = os.listdir(args.source)
    mapping_list = [p for p in mapping_list if p.startswith('wrap_') and not (p.endswith('.tar'))] 
    mapping_list.sort(key=lambda x: int(x.split('_')[1]))
    # mapping_list = [wrap_0, wrap_1, ...]
    
    for wrap_dir in mapping_list:
        # wrap_dir = "wrap_*"
        image_pairs = find_image_pairs(os.path.join(args.source, wrap_dir))[:2]
        # N * (gt_path, wrap_path, frame_id, cam_id)
        
        if not image_pairs:
            print(f"在 {wrap_dir} 中未找到匹配的图片对")
            return
        
        
       
        # 处理每个图片对
        for gt_path, wrap_path, frame_id, cam_id in image_pairs:
            print(f"\n处理 Mapping {wrap_dir.split('_')[1]} - Wrap {frame_id}- Cam {cam_id}")

            # 初始化跟踪器（每个图片对使用新的跟踪器）
            tracker = BotSort(
                reid_weights=Path(args.reid_weights),
                device=device,
                half=False,
            )
            
            # 处理GT图片（第一帧）
            gt_dets, gt_masks, gt_image = process_image(gt_path, segmentation_model, device, args.conf_thres)
            if gt_image is None:
                continue
            
            # 更新跟踪器 - GT帧
            gt_tracks = tracker.update(gt_dets, gt_image)
            
            # 过滤小边界框
            gt_tracks_filtered = filter_small_bboxes(gt_tracks, args.min_bbox_area)
            
            # # 根据类别平均面积进一步过滤
            # if len(gt_tracks_filtered) > 0:
            #     print(f"GT帧跟踪结果过滤:")
            #     gt_tracks_filtered = filter_by_class_average_area(gt_tracks_filtered, threshold_ratio=0.5)
            
            gt_track_mask, gt_track_info = create_tracking_mask(gt_tracks_filtered, gt_masks, gt_image.shape)
            
            # 处理Wrap图片（第二帧）
            wrap_dets, wrap_masks, wrap_image = process_image(wrap_path, segmentation_model, device, args.conf_thres)
            if wrap_image is None:
                continue
            
            # 更新跟踪器 - Wrap帧
            wrap_tracks = tracker.update(wrap_dets, wrap_image)
            
            # 过滤小边界框
            wrap_tracks_filtered = filter_small_bboxes(wrap_tracks, args.min_bbox_area)
            
            # 根据类别平均面积进一步过滤
            if len(wrap_tracks_filtered) > 0:
                print(f"Wrap帧跟踪结果过滤:")
                wrap_tracks_filtered = filter_by_class_average_area(wrap_tracks_filtered, threshold_ratio=0.5)
            
            wrap_track_mask, wrap_track_info = create_tracking_mask(wrap_tracks_filtered, wrap_masks, wrap_image.shape)
            
            # 计算运动强烈程度
            gt_ids = set(gt_track_info.keys())
            wrap_ids = set(wrap_track_info.keys())
            common_ids = gt_ids & wrap_ids
            rgb_diff = np.abs(gt_image.astype(np.float32) - wrap_image.astype(np.float32)).mean(axis=2)  # (H,W,3)=> (H,W)
            thre  = np.percentile(rgb_diff, 98)  # 取95%分位数作为阈值
            
            # 创建二值掩码并进行腐蚀操作去除噪音
            binary_mask = (rgb_diff >= thre).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3x3椭圆形核
            binary_mask_eroded = cv2.erode(binary_mask, kernel, iterations=1)
            
            # 将腐蚀后的掩码应用到rgb_diff
            # rgb_diff = rgb_diff * binary_mask_eroded
            
            if common_ids:
                # print(f"在两帧都出现的ID: {sorted(common_ids)}")
                
                # 计算每个共同ID的运动指标
                pair_metrics = {}
                pair_metrics_count = {}
                for track_id in common_ids:
                    # 获取两帧中该ID的掩码
                    gt_mask = get_mask_by_track_id(gt_tracks_filtered, gt_masks, track_id)
                    wrap_mask = get_mask_by_track_id(wrap_tracks_filtered, wrap_masks, track_id)

                    gt_bbox = get_bbox_by_track_id(gt_tracks_filtered, track_id)
                    wrap_bbox = get_bbox_by_track_id(wrap_tracks_filtered, track_id)


                   
                    
                    x1_1, y1_1, x2_1, y2_1 = gt_bbox
                    x1_2, y1_2, x2_2, y2_2 = wrap_bbox

                    # 去除突变情况
                    # Boundbox突变，不进行动态积分计算
                    horizon_change_ratio = abs((x2_2 - x1_2) - (x2_1 - x1_1))/abs(x2_1 - x1_1)
                    vertical_change_ratio = abs((y2_2 - y1_2) - (y2_1 - y1_1))/abs(y2_1 - y1_1)
                    if horizon_change_ratio > 0.3 or vertical_change_ratio > 0.3:
                        print(f"  ID {track_id}: 边界框突变，跳过计算")
                        continue

                    motion_intensity_1 = calculate_motion_metrics(gt_bbox, wrap_bbox)
                    # if motion_intensity_1 is not None:
                    #     pair_metrics[track_id] = motion_intensity_1
                    #     pair_metrics_count[track_id] = pair_metrics_count.get(track_id, 0) + 1


                    
                    if gt_mask is not None and wrap_mask is not None:
                        # 二值化掩码
                        gt_binary_mask = (gt_mask > 0.5).astype(np.uint8)
                        wrap_binary_mask = (wrap_mask > 0.5).astype(np.uint8)
                        
                        # 计算掩码面积
                        gt_mask_area = np.sum(gt_binary_mask)
                        wrap_mask_area = np.sum(wrap_binary_mask)
                        
                        if gt_mask_area > 0 and wrap_mask_area > 0:
                            # 计算掩码内的RGB差异
                            gt_mask_diff = binary_mask_eroded[gt_binary_mask == 1].sum()/ gt_mask_area
                            wrap_mask_diff = binary_mask_eroded[wrap_binary_mask == 1].sum()/ wrap_mask_area
                            motion_intensity = gt_mask_diff + wrap_mask_diff
                          
                            
                            if motion_intensity is not None:
                                if motion_intensity_1 is not None:
                                    motion_intensity += motion_intensity_1
                                pair_metrics[track_id] = motion_intensity
                                pair_metrics_count[track_id] = pair_metrics_count.get(track_id, 0) + 1
                        else:
                            print(f"  ID {track_id}: 掩码面积为0")
                    else:
                        print(f"  ID {track_id}: 无法获取掩码")
                

            # ID matching
            seq_mask_paths = os.listdir(args.all_seq_mask_path)
            seq_mask_paths = sorted([p for p in seq_mask_paths if p.endswith('.npy')])
            seq_mask_path = seq_mask_paths[int(frame_id)]
            print(f"Matching wrap_{frame_id} back to seq_mask: {seq_mask_path}")
            # 创建ID映射
            all_seq_mask = np.load(os.path.join(args.all_seq_mask_path, seq_mask_path))
            all_seq_mask = cv2.resize(all_seq_mask, (gt_track_mask.shape[1], gt_track_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            id_mapping = create_id_mapping(all_seq_mask, gt_track_mask)
            # 更新当前wrap帧动态得分
            pair_metrics_updated = {}
            pair_metrics_count_updated = {}
            for track_id, motion_intensity in pair_metrics.items():
                if track_id in id_mapping:
                    mapped_id = id_mapping[track_id]
                    pair_metrics_updated[mapped_id] = motion_intensity
                    pair_metrics_count_updated[mapped_id] = pair_metrics_count.get(track_id, 0)
                    # print(f"ID {track_id} 映射到 {mapped_id}，运动强度: {motion_intensity:.4f}")
                else:
                    print(f"ID {track_id} 没有映射到任何ID")
            # 更新seq_ID_score
            for track_id, motion_intensity in pair_metrics_updated.items():
                if track_id not in seq_ID_score:
                    seq_ID_score[track_id] = motion_intensity
                    seq_ID_score_count[track_id] = pair_metrics_count_updated.get(track_id, 0)
                else:
                    seq_ID_score[track_id] += motion_intensity
                    seq_ID_score_count[track_id] += pair_metrics_count_updated.get(track_id, 0)

            # 可视化跟踪结果
            # 将 rgb_diff 转换为可视化图像
            rgb_diff_vis = (rgb_diff * 255 / np.max(rgb_diff)).astype(np.uint8)
            rgb_diff_vis = cv2.applyColorMap(rgb_diff_vis, cv2.COLORMAP_JET)
            rgb_diff_vis = cv2.resize(rgb_diff_vis, (gt_image.shape[1], gt_image.shape[0]))

            gt_vis, gt_diff_vis = visualize_tracking_result(gt_image, gt_tracks_filtered, gt_masks, id_mapping, rgb_diff_vis)
            wrap_vis, wrap_diff_vis = visualize_tracking_result(wrap_image, wrap_tracks_filtered, wrap_masks, id_mapping, rgb_diff_vis)
            
            # 使用matplotlib创建2x2子图布局
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # 将BGR图像转换为RGB（matplotlib使用RGB格式）
            gt_vis_rgb = cv2.cvtColor(gt_vis, cv2.COLOR_BGR2RGB)
            wrap_vis_rgb = cv2.cvtColor(wrap_vis, cv2.COLOR_BGR2RGB)
            gt_diff_vis_rgb = cv2.cvtColor(gt_diff_vis, cv2.COLOR_BGR2RGB)
            wrap_diff_vis_rgb = cv2.cvtColor(wrap_diff_vis, cv2.COLOR_BGR2RGB)
            
            # 第一行：GT和Wrap的跟踪结果
            axes[0, 0].imshow(gt_vis_rgb)
            axes[0, 0].set_title(f'GT Frame {frame_id}', fontsize=14, fontweight='bold')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(wrap_vis_rgb)
            axes[0, 1].set_title(f'Wrap Frame {frame_id}', fontsize=14, fontweight='bold')
            axes[0, 1].axis('off')
            
            # 第二行：GT和Wrap的差分图
            axes[1, 0].imshow(gt_diff_vis_rgb)
            axes[1, 0].set_title(f'GT Diff Frame {frame_id}', fontsize=14, fontweight='bold')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(wrap_diff_vis_rgb)
            axes[1, 1].set_title(f'Wrap Diff Frame {frame_id}', fontsize=14, fontweight='bold')
            axes[1, 1].axis('off')
            
            # 调整子图间距
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(output_path / wrap_dir, exist_ok=True)
            combined_vis_path = output_path / wrap_dir / f"wrap_{frame_id}_cam{cam_id}_mask.png"
            plt.savefig(str(combined_vis_path), dpi=150, bbox_inches='tight')
            plt.close()  # 关闭图形以释放内存
            
            print(f"保存2x2子图可视化结果到 {combined_vis_path}")

        #打印最终的ID得分
        print(f"\n================= 结束 mapping{wrap_dir.split('_')[1]} ID得分 =================")
        temp_scores = [] # 存储临时得分列表
        for track_id, score in sorted(seq_ID_score.items()):
            temp_scores.append((track_id, score / seq_ID_score_count.get(track_id, 1)))
        temp_scores.sort(key=lambda x: x[1], reverse=True)
        for track_id, score in temp_scores:
            print(f"ID {track_id}: 动态得分 {score:.4f} (计数: {seq_ID_score_count.get(track_id, 1)})")

        # 保存跟踪结果
        os.makedirs(output_path / wrap_dir, exist_ok=True)
        tmp_score_file = output_path / wrap_dir / "dynamic_scores.txt"
        with open(tmp_score_file, 'w') as f:
            f.write("ID\t动态得分\n")
            for track_id, score in temp_scores:
                f.write(f"{track_id}\t{score:.4f}\n")

       

        

    print(f"\n================= 最终ID得分 =================")
    final_scores = []  # 存储最终得分列表
    for track_id, score in sorted(seq_ID_score.items()):
        if seq_ID_score_count.get(track_id, 1) > 6:
            final_scores.append((track_id, score / seq_ID_score_count.get(track_id, 1)))
    final_scores.sort(key=lambda x: x[1], reverse=True)
    for track_id, score in final_scores:
        print(f"ID {track_id}: 动态得分 {score:.4f} (计数: {seq_ID_score_count.get(track_id, 1)})")
    
    # 将分数字典保存到文件
    score_file = output_path / "dynamic_scores.txt"
    with open(score_file, 'w') as f:
        f.write("ID\t动态得分\n")
        for track_id, score in final_scores:
            f.write(f"{track_id}\t{score:.4f}\n")
    print(f"动态得分已保存到 {score_file}")

    # # 将得分从高到低可视化为折线图
    # import matplotlib.pyplot as plt
    # ids = list(seq_ID_score.keys())
    # scores = list(seq_ID_score.values())
    # plt.figure(figsize=(12, 6))
    # plt.plot(ids, scores, marker='o')
    # plt.title('动态得分')
    # plt.xlabel('ID')
    # plt.ylabel('动态得分')
    # plt.xticks(rotation=45)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(output_path / "dynamic_scores_plot.png")


if __name__ == "__main__":
    main()
    # 从输出路径加载txt为得分字典
    # 例如：seq_ID_score = {'1': 0.8, '2': 0.6, ...}

# python image_pair_tracking.py --source wrap_0017085 --output wrap_0017085_output --device cpu --all_seq_mask_path tracking_masks/npy_masks_0017085