import lap
import numpy as np
import os
import argparse
from scipy.spatial.distance import cdist

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b

def embedding_distance(view_A, view_B, metric='cosine'):
    """
    Compute cost based on feature distance
    :type view_A: np.ndarray (N, dim) 
    :type view_B: np.ndarray (M, dim)
    :type metric: str, 'cosine' or 'euclidean'

    :rtype cost_matrix np.ndarray
    """
    if len(view_A) == 0 or len(view_B) == 0:
        return np.zeros((len(view_A), len(view_B)), dtype=np.float32)

    if (len(view_A) > 0 and isinstance(view_A[0], np.ndarray)) or (
        len(view_B) > 0 and isinstance(view_B[0], np.ndarray)
    ):
        features_A = view_A
        features_B = view_B
    else:
        features_A = [track.curr_feat for track in view_A]
        features_B = [track.curr_feat for track in view_B]
    
    if metric == 'cosine':
        # 余弦距离转换为相似度代价矩阵
        # cosine_distances返回的是1-余弦相似度，值域[0,2]
        # 我们返回的是余弦距离，与原有逻辑兼容（值越小表示越相似）
        cost_matrix = cdist(np.asarray(features_A), np.asarray(features_B), metric)
    else:
        # 保持原有的欧氏距离计算方式
        cost_matrix = cdist(np.asarray(features_A), np.asarray(features_B), metric)
    
    return cost_matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="跨视角ReID")
    parser.add_argument('--view_A', type=str, required=True, help='视角A路径')
    parser.add_argument('--view_B', type=str, required=True, help='视角B路径')
    parser.add_argument('--use_ema', action='store_true', help='是否使用EMA特征')
    parser.add_argument('--dynamic_list_A', type=str, default="", help='视角A的动态ID列表，逗号分隔')
    parser.add_argument('--dynamic_list_B', type=str, default="", help='视角B的动态ID列表，逗号分隔')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'], help='特征距离度量方式: cosine或euclidean')
    args = parser.parse_args()

    if args.use_ema:
        view_A_path = os.path.join(args.view_A, "ema_feat.npy")
        view_B_path = os.path.join(args.view_B, "ema_feat.npy")
    else:
        view_A_path = os.path.join(args.view_A, "instant_feat.npy")
        view_B_path = os.path.join(args.view_B, "instant_feat.npy")

    record_A = np.load(os.path.join(args.view_A, "frame_record.npy"))
    record_B = np.load(os.path.join(args.view_B, "frame_record.npy"))
    frame_record_A_o = record_A[:,:2]
    frame_record_B_o = record_B[:,:2]
    area_record_A_o = record_A[:,2]
    area_record_B_o = record_B[:,2]
    cls_record_A_o = record_A[:,3]
    cls_record_B_o = record_B[:,3]
    view_A_features_o = np.load(view_A_path)
    view_B_features_o = np.load(view_B_path)

    # 取出对应行

    dynamic_A = [int(i) for i in args.dynamic_list_A.split(',') if i.strip().isdigit()]
    dynamic_B = [int(i) for i in args.dynamic_list_B.split(',') if i.strip().isdigit()]
    print(f"Dynamic IDs in view A: {dynamic_A}")
    print(f"Dynamic IDs in view B: {dynamic_B}")
    dynamic_A = [i-1 for i in dynamic_A]
    dynamic_B = [i-1 for i in dynamic_B]

    view_A_features = view_A_features_o[dynamic_A]
    view_B_features = view_B_features_o[dynamic_B]
    frame_record_A = frame_record_A_o[dynamic_A]
    frame_record_B = frame_record_B_o[dynamic_B]
    area_record_A = area_record_A_o[dynamic_A]
    area_record_B = area_record_B_o[dynamic_B]
    cls_record_A = cls_record_A_o[dynamic_A]
    cls_record_B = cls_record_B_o[dynamic_B]

    # print(f"frame_record_A: {frame_record_A}")
    # print(f"frame_record_B: {frame_record_B}")
    # print(f"area_record_A: {area_record_A}")
    # print(f"area_record_B: {area_record_B}")



    # 计算帧区间间隔
    MAX_TIME_GAP = 30 # 最多间隔30帧
    MAX_TIME_OVERLAP = -20 # 最多共享20帧
    time_gap_matrix = np.zeros((len(frame_record_A), len(frame_record_B)), dtype=np.int32)
    for i, (start_A, end_A) in enumerate(frame_record_A):
        for j, (start_B, end_B) in enumerate(frame_record_B):
           time_gap_matrix[i, j] = max(start_B - end_A, start_A - end_B)

    print("Time gap matrix:")
    print(time_gap_matrix)

    # 计算面积比例
    MIN_AREA_RATIO = 0.05
    area_matrix = np.zeros((len(area_record_A), len(area_record_B)), dtype=np.float32)
    for i, area_A in enumerate(area_record_A):
        for j, area_B in enumerate(area_record_B):
            area_ratio = area_A / area_B if area_B > 0 else 1.0
            area_matrix[i, j] = min(area_ratio, 1.0 / area_ratio)  # 取较小值，确保比例在0到1之间
    

    # print("Area ratio matrix:")
    # print(area_matrix)
    
    # 计算类别矩阵，相同类别为1，不同类别为0
    cls_matrix = np.zeros((len(cls_record_A), len(cls_record_B)), dtype=np.int32)
    for i, cls_A in enumerate(cls_record_A):
        for j, cls_B in enumerate(cls_record_B):
            cls_matrix[i, j] = 1 if cls_A == cls_B and cls_A != -1 else 0  # -1表示无效类别，不匹配


    # 计算外观特征距离矩阵
    MAX_FEATURE_DIST = 0.5
    emb_dists = embedding_distance(view_A_features, view_B_features, metric=args.metric)
    emb_dists = emb_dists / 2.0

    print(f"Initial embedding distance matrix (using {args.metric} metric):")
    print(emb_dists)

    # 综合考虑时间、面积、类别等因素，调整距离矩阵
    emb_dists[emb_dists > MAX_FEATURE_DIST] = 1.0
    emb_dists[time_gap_matrix > MAX_TIME_GAP] = 1.0  # 超过时间间隔的，距离设为1.0
    emb_dists[time_gap_matrix < MAX_TIME_OVERLAP] = 1.0  # 共享时间过多的，距离设为1.0
    emb_dists[area_matrix < MIN_AREA_RATIO] = 1.0  # 面积比例过大的，距离设为1.0
    emb_dists[cls_matrix == 0] = 1.0  # 类别不匹配的，距离设为1.0
    
    print("Final distance matrix:")
    print(emb_dists)

    matches, u_a, u_b = linear_assignment(emb_dists, 0.8)

        

    # 将matches映射回原始ID
    mapped_matches = [(dynamic_A[m[0]] + 1, dynamic_B[m[1]] + 1) for m in matches]
    

    print(f"Matches: {matches.shape}, Unmatched A: {len(u_a)}, Unmatched B: {len(u_b)}")
    print(f"Matched pairs (view_A index, view_B index):\n{matches}")
    print(f"Mapped Matches (view_A ID, view_B ID):\n{mapped_matches}")
    
    print(f"Frame records of matches:")
    for m in mapped_matches:
        A_start, A_end = frame_record_A_o[m[0]-1]
        B_start, B_end = frame_record_B_o[m[1]-1]
        print(f"ID {m[0]} (Frames {A_start}-{A_end}) <-> ID {m[1]} (Frames {B_start}-{B_end})")
    
    print(f"Area records of matches:")
    for m in mapped_matches:
        area_A = area_record_A_o[m[0]-1]
        area_B = area_record_B_o[m[1]-1]
        area_ratio = area_A/area_B if area_A/area_B<1 else area_B/area_A
        print(f"ID {m[0]} Area: {area_A:.2f} <-> ID {m[1]} Area: {area_B:.2f}   Ratio:{area_ratio:.4f}")

