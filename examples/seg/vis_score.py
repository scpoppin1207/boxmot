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

def visualize_dynamic_scores(score_dict, threshold=None, output_path=None, star_ids=None):
    """可视化动态得分为折线图
    
    Args:
        score_dict: {ID: score} 字典
        threshold: 阈值线（外部计算好的）
        output_path: 输出图片路径，如果为None则只显示
        star_ids: 要显示为五角星的ID列表
    """
    # 如果未提供star_ids，使用默认值
    if star_ids is None:
        star_ids = [1, 17, 29]
    
    # 按得分从高到低排序
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    ids = [item[0] for item in sorted_items][:10]
    scores = [item[1] for item in sorted_items][:10]
    scores = [x*x*x for x in scores]  # 对得分进行立方处理
    
    # 创建单图布局，设置为正方形图
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # 画出所有点
    for i in range(len(ids)):
        if ids[i] in star_ids:
            # 使用五角星标记特定ID
            ax.plot(i, scores[i], marker='*', markersize=25, color='red', 
                   markerfacecolor='gold', markeredgecolor='red', linestyle='none')
        else:
            # 使用圆形标记其他ID
            ax.plot(i, scores[i], marker='o', markersize=15, 
                   color='#1f77b4', markerfacecolor='#ff7f0e', markeredgecolor='#1f77b4')
    
    # 连接所有点形成折线
    ax.plot(range(len(ids)), scores, linewidth=3, color='#1f77b4', alpha=0.7)
    
    # 可视化阈值线
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Threshold: {threshold:.2e}')
        
        # # 添加阈值标注
        # ax.annotate(f'Threshold\n{threshold:.2e}', 
        #             xy=(len(ids)*0.8, threshold),
        #             xytext=(len(ids)*0.8, threshold + max(scores)*0.1),
        #             arrowprops=dict(arrowstyle='->', color='red', lw=2),
        #             fontsize=10, fontweight='bold',
        #             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        # # 添加图例
        # ax.legend(loc='upper right')
    
    # 设置图的标题和标签
    # ax.set_title('Dynamic Motion Scores by Track ID', fontsize=16, fontweight='bold')
    ax.set_xlabel('Track ID Rank ', fontsize=20)
    ax.set_ylabel('Score Value', fontsize=20)
    
    # 设置x轴标签为实际的ID
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([str(id_) for id_ in ids], rotation=45, ha='right')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # # 在图上显示具体数值
    # for i in range(len(ids)):
    #     # 为五角星ID添加特殊标注
    #     if ids[i] in star_ids:
    #         ax.annotate(f'ID:{ids[i]}\n{scores[i]:.2e}', 
    #                     (i, scores[i]), 
    #                     textcoords="offset points", 
    #                     xytext=(0,10), 
    #                     ha='center', 
    #                     fontsize=9,
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='gold', alpha=0.8))
    #     elif i % 3 ==0:
            
    #         ax.annotate(f'{scores[i]:.2e}', 
    #                     (i, scores[i]), 
    #                     textcoords="offset points", 
    #                     xytext=(0,10), 
    #                     ha='center', 
    #                     fontsize=8,
    #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    # 设置严格的正方形比例
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dynamic scores plot saved to: {output_path}")
    
    # 打印统计信息（英文）
    print(f"\nStatistics:")
    print(f"Total number of tracked objects: {len(ids)}")
    print(f"Highest dynamic score: {max(scores):.4f} (ID: {ids[0]})")
    print(f"Lowest dynamic score: {min(scores):.4f} (ID: {ids[-1]})")
    print(f"Average dynamic score: {np.mean(scores):.4f}")
    print(f"Standard deviation: {np.std(scores):.4f}")
    
    # 打印阈值信息
    if threshold is not None:
        print(f"Threshold: {threshold:.4f}")
        # 统计高于阈值的对象数量
        above_threshold = [score for score in scores if score >= threshold]
        print(f"Objects above threshold: {len(above_threshold)}/{len(scores)}")
    
    # 打印标为五角星的ID
    star_ids_present = [id_ for id_ in ids if id_ in star_ids]
    if star_ids_present:
        print(f"\nIDs visualized as stars: {star_ids_present}")
    
    print(f"\nTop 10 most dynamic objects:")
    for i in range(min(10, len(ids))):
        print(f"  Rank {i+1}: ID {ids[i]} - Score {scores[i]:.4f}")

if __name__ == "__main__":
    # 设置文件路径
    parser = argparse.ArgumentParser(description="图片对分割跟踪")
    parser.add_argument('--source_file', type=str, required=True, help='字典文件路径')
    parser.add_argument('--star_ids', type=str, default="1,17,29", help='要标记为五角星的ID，用逗号分隔')
    args = parser.parse_args()

    score_file = os.path.join(args.source_file, "dynamic_scores.txt")
    output_dir = args.source_file
    
    # 解析要标记为五角星的ID
    star_ids = [int(id_.strip()) for id_ in args.star_ids.split(',')]
    print(f"IDs to be marked as stars: {star_ids}")

    # 加载动态得分字典
    print("Loading dynamic scores...")
    score_dict = load_dynamic_scores(score_file)
    if len(score_dict) == 0:
        print("No scores found in the file, exiting.")
        exit(1)
        
    threshold = 1e-3
    
    if score_dict:
        print(f"Successfully loaded {len(score_dict)} track IDs with scores")
        
        # 创建折线图，使用1:1比例
        line_plot_path = os.path.join(output_dir, "dynamic_scores_single_plot_quadra.png") 
        visualize_dynamic_scores(score_dict, threshold, str(line_plot_path), star_ids)
        
    else:
        print("No scores found in the file")
    
