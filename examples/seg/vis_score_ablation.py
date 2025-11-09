#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from pathlib import Path
import argparse
import os

def load_dynamic_scores(file_path):
    """从txt文件加载动态得分字典
    
    Args:
        file_path: 动态得分文件路径
    
    Returns:
        score_dict: {ID: score} 字典
    """
    score_dict = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # 跳过标题行
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split('\t')
            if len(parts) == 2:
                track_id = int(parts[0])
                score = float(parts[1])
                score_dict[track_id] = score
    
    return score_dict

def visualize_dynamic_scores(score_dict, score_dict_exp1=None, score_dict_exp2=None, threshold=None, output_path=None, star_ids=None):
    """可视化动态得分为折线图，同时比较不同的实验结果和指数处理
    
    Args:
        score_dict: 主要实验的 {ID: score} 字典
        score_dict_exp1: 实验1的 {ID: score} 字典（可选）
        score_dict_exp2: 实验2的 {ID: score} 字典（可选）
        threshold: 阈值线（外部计算好的）
        output_path: 输出图片路径，如果为None则只显示
        star_ids: 要显示为五角星的ID列表
    """
    # 如果未提供star_ids，使用默认值
    if star_ids is None:
        star_ids = [1, 17, 29]
    
    # 处理主要实验数据
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    ids = [item[0] for item in sorted_items][:10]
    raw_scores = [item[1] for item in sorted_items][:10]
    scores_squared = [x*x for x in raw_scores]  # 平方处理
    scores_cubed = [x*x*x for x in raw_scores]  # 立方处理
    socres_exp = [np.exp(x) for x in raw_scores]  # 指数处理
    
    # 处理实验1数据(如果有)
    exp1_ids = []
    exp1_raw_scores = []
    exp1_scores_squared = []
    exp1_scores_cubed = []
    if score_dict_exp1:
        exp1_sorted_items = sorted(score_dict_exp1.items(), key=lambda x: x[1], reverse=True)
        exp1_ids = [item[0] for item in exp1_sorted_items][:10]
        exp1_raw_scores = [item[1] for item in exp1_sorted_items][:10]
        exp1_scores_cubed = [x*x*x for x in exp1_raw_scores]  # 立方处理
    
    # 处理实验2数据(如果有)
    exp2_ids = []
    exp2_raw_scores = []
    exp2_scores_squared = []
    exp2_scores_cubed = []
    if score_dict_exp2:
        exp2_sorted_items = sorted(score_dict_exp2.items(), key=lambda x: x[1], reverse=True)
        exp2_ids = [item[0] for item in exp2_sorted_items][:10]
        exp2_raw_scores = [item[1] for item in exp2_sorted_items][:10]
        exp2_scores_cubed = [x*x*x for x in exp2_raw_scores]  # 立方处理
    
    # 创建2x3子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 创建图表标题
    titles = [
        ["Cube", "Square", "Raw"],
        ["Motion Difference Only", "Appearance Difference Only", ""]
    ]
    
    # 定义数据集
    data_sets = [
        [
            {"ids": ids, "scores": scores_cubed, "title": "Cube", "color": "#1f77b4", "power": 3},
            {"ids": exp1_ids, "scores": scores_squared, "title": "Square", "color": "#ff7f0e", "power": 2},
            {"ids": exp2_ids, "scores": raw_scores, "title": "Raw", "color": "#2ca02c", "power": 1},
        ],
        [
            {"ids": ids, "scores": socres_exp, "title": "Exp", "color": "#f2ff00da", "power": 0},
            {"ids": ids, "scores": exp1_scores_cubed, "title": "Position Inconsistency Only", "color": "#00d9ff", "power": 3},
            {"ids": ids, "scores": exp2_scores_cubed, "title": "Appearance Inconsistenty Only", "color": "#ff0000", "power": 3},
        ]
    ]
    
    # 定义子图标签
    subplot_labels = [
        ['(a)', '(b)', '(c)'],
        ['(d)', '(e)', '(f)']
    ]
    
    # 遍历绘制每个子图
    for row in range(2):
        for col in range(3):
            # # 跳过空白子图
            # if row == 1 and col == 0:
            #     axes[row, col].axis('off')  # 关闭坐标轴显示
            #     continue
                
            ax = axes[row, col]
            data = data_sets[row][col]
            
            if not data or not data["ids"] or not data["scores"]:
                ax.text(0.5, 0.5, "No Data Available", 
                       ha='center', va='center', fontsize=14, 
                       transform=ax.transAxes)
                continue
                
            curr_ids = data["ids"]
            curr_scores = data["scores"]
            
            # 绘制所有点
            for i in range(len(curr_ids)):
                if curr_ids[i] in star_ids:
                    # 使用五角星标记特定ID
                    ax.plot(i, curr_scores[i], marker='*', markersize=25, color='red', 
                           markerfacecolor='gold', markeredgecolor='red', linestyle='none')
                else:
                    # 使用圆形标记其他ID
                    ax.plot(i, curr_scores[i], marker='o', markersize=15, 
                           color=data["color"], markerfacecolor='#ff7f0e', markeredgecolor=data["color"])
            
            # 连接所有点形成折线
            ax.plot(range(len(curr_ids)), curr_scores, linewidth=5, color=data["color"], alpha=0.7)
            
                # 在所有子图中使用相同的阈值线
            if threshold is not None:
                # 直接使用传入的阈值，不应用任何幂次
                threshold_value = threshold
                threshold_label = f' '
                # 绘制阈值线
                ax.axhline(y=threshold_value, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                
                # 添加阈值标注
                y_min, y_max = ax.get_ylim()
                # 计算合适的标注位置
                annotation_y = threshold_value * 1.2
                if annotation_y > y_max * 0.9:  # 如果标注位置太高
                    annotation_y = threshold_value * 0.8
                
                # 添加阈值标注在图的右侧
                
                
                # # 显示图例
                # ax.legend(loc='upper right', fontsize=8)
            
            # 设置图的标题和标签
            ax.set_title(data["title"], fontsize=20, fontweight='bold')
            ax.set_xlabel('Track ID Rank', fontsize=20)
            ax.set_ylabel('Score Value', fontsize=20)
            
            # 添加子图标签 (a) 到 (f)
            ax.text(0.03, 0.95, subplot_labels[row][col], 
                   transform=ax.transAxes, 
                   fontsize=16, fontweight='bold', 
                   verticalalignment='top', 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
            
            # 关闭x轴和y轴的数字标注
            ax.set_xticks([])  # 清空x轴刻度
            ax.set_yticks([])  # 清空y轴刻度
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--')
    # 设置子图间距和整体布局
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.10, wspace=0.2, hspace=0.3)
    
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dynamic scores comparison plot saved to: {output_path}")
    
    # 打印主实验的统计信息
    print(f"\nMain Experiment Statistics:")
    print(f"Total number of tracked objects: {len(ids)}")
    print(f"Highest dynamic score: {max(raw_scores):.4e} (ID: {ids[0]})")
    print(f"Lowest dynamic score: {min(raw_scores):.4e} (ID: {ids[-1]})")
    print(f"Average dynamic score: {np.mean(raw_scores):.4e}")
    print(f"Standard deviation: {np.std(raw_scores):.4e}")
    
    # 打印阈值信息
    if threshold is not None:
        print(f"Threshold: {threshold:.4e}")
        # 统计高于阈值的对象数量（对所有分数类型分别计算）
        above_threshold_raw = [score for score in raw_scores if score >= threshold]
        above_threshold_squared = [score for score in scores_squared if score >= threshold]
        above_threshold_cubed = [score for score in scores_cubed if score >= threshold]
        
        print(f"Raw scores above threshold: {len(above_threshold_raw)}/{len(raw_scores)}")
        print(f"Squared scores above threshold: {len(above_threshold_squared)}/{len(scores_squared)}")
        print(f"Cubed scores above threshold: {len(above_threshold_cubed)}/{len(scores_cubed)}")
    
    # 打印标为五角星的ID
    star_ids_present = [id_ for id_ in ids if id_ in star_ids]
    if star_ids_present:
        print(f"\nIDs visualized as stars: {star_ids_present}")
    
    print(f"\nTop 10 most dynamic objects (main experiment):")
    for i in range(min(10, len(ids))):
        print(f"  Rank {i+1}: ID {ids[i]} - Score (raw): {raw_scores[i]:.4e}, (squared): {scores_squared[i]:.4e}, (cubed): {scores_cubed[i]:.4e}")
        
    # 如果有对照实验，打印相应信息
    if score_dict_exp1 and exp1_ids:
        print(f"\nExperiment 1 top scores:")
        for i in range(min(5, len(exp1_ids))):
            print(f"  Rank {i+1}: ID {exp1_ids[i]} - Score (cubed): {exp1_scores_cubed[i]:.4e}")
            
    if score_dict_exp2 and exp2_ids:
        print(f"\nExperiment 2 top scores:")
        for i in range(min(5, len(exp2_ids))):
            print(f"  Rank {i+1}: ID {exp2_ids[i]} - Score (cubed): {exp2_scores_cubed[i]:.4e}")

if __name__ == "__main__":
    # 设置文件路径
    parser = argparse.ArgumentParser(description="动态得分可视化比较工具")
    parser.add_argument('--source_file', type=str, required=True, help='主要实验的字典文件路径')
    parser.add_argument('--exp1_file', type=str, help='对照实验1的字典文件路径')
    parser.add_argument('--exp2_file', type=str, help='对照实验2的字典文件路径')
    parser.add_argument('--star_ids', type=str, default="1,17,29", help='要标记为五角星的ID，用逗号分隔')
    parser.add_argument('--threshold', type=float, default=1e-3, help='阈值')
    parser.add_argument('--output', type=str, help='输出文件名，默认为比较结果图')
    args = parser.parse_args()

    # 主实验数据路径
    score_file = os.path.join(args.source_file, "dynamic_scores.txt")
    output_dir = args.source_file
    
    # 解析要标记为五角星的ID
    star_ids = [int(id_.strip()) for id_ in args.star_ids.split(',')]
    print(f"IDs to be marked as stars: {star_ids}")
    
    # 设置阈值
    threshold = args.threshold
    
    # 加载主实验动态得分字典
    print(f"Loading main experiment scores from {score_file}...")
    score_dict = load_dynamic_scores(score_file)
    if len(score_dict) == 0:
        print("No scores found in the main experiment file, exiting.")
        exit(1)
    print(f"Successfully loaded {len(score_dict)} track IDs from main experiment")
    
    # 加载对照实验1数据（如果提供）
    score_dict_exp1 = None
    if args.exp1_file:
        exp1_score_file = os.path.join(args.exp1_file, "dynamic_scores.txt")
        print(f"Loading experiment 1 scores from {exp1_score_file}...")
        score_dict_exp1 = load_dynamic_scores(exp1_score_file)
        if score_dict_exp1:
            print(f"Successfully loaded {len(score_dict_exp1)} track IDs from experiment 1")
        else:
            print("Warning: No data found for experiment 1")
    
    # 加载对照实验2数据（如果提供）
    score_dict_exp2 = None
    if args.exp2_file:
        exp2_score_file = os.path.join(args.exp2_file, "dynamic_scores.txt")
        print(f"Loading experiment 2 scores from {exp2_score_file}...")
        score_dict_exp2 = load_dynamic_scores(exp2_score_file)
        if score_dict_exp2:
            print(f"Successfully loaded {len(score_dict_exp2)} track IDs from experiment 2")
        else:
            print("Warning: No data found for experiment 2")
    
    # 设置输出文件路径
    output_filename = args.output if args.output else "dynamic_scores_comparison.png"
    line_plot_path = os.path.join(output_dir, output_filename)
    
    # 创建比较图
    visualize_dynamic_scores(
        score_dict, 
        score_dict_exp1, 
        score_dict_exp2, 
        threshold, 
        str(line_plot_path), 
        star_ids
    )
    
