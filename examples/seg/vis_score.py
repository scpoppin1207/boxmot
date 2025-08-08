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

def visualize_dynamic_scores(score_dict, threshold=None, output_path=None):
    """可视化动态得分为折线图
    
    Args:
        score_dict: {ID: score} 字典
        threshold: 阈值线（外部计算好的）
        output_path: 输出图片路径，如果为None则只显示
    """
    # 按得分从高到低排序
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    ids = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    
    # 创建图形
    plt.figure(figsize=(14, 8))
    
    # 绘制折线图
    plt.plot(range(len(ids)), scores, marker='o', linewidth=2, markersize=6, 
             color='#1f77b4', markerfacecolor='#ff7f0e', markeredgecolor='#1f77b4')
    
    # 可视化阈值线
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, linewidth=2,
                   label=f'Threshold: {threshold:.4f}')
        
        # 添加阈值标注
        plt.annotate(f'Threshold\n{threshold:.4f}', 
                    xy=(len(ids)*0.8, threshold),
                    xytext=(len(ids)*0.8, threshold + max(scores)*0.1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.8))
        
        # 添加图例
        plt.legend(loc='upper right')
    
    # 设置标题和标签（只使用英文）
    plt.title('Dynamic Motion Scores by Track ID', fontsize=16, fontweight='bold')
    plt.xlabel('Track ID Rank (Sorted by Score)', fontsize=12)
    plt.ylabel('Dynamic Motion Score', fontsize=12)
    
    # 设置x轴标签为实际的ID
    plt.xticks(range(len(ids)), [str(id_) for id_ in ids], rotation=45, ha='right')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 在图上显示具体数值（只显示前10个最高分）
    for i in range(min(10, len(ids))):
        plt.annotate(f'{scores[i]:.2f}', 
                    (i, scores[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 调整布局
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
    
    print(f"\nTop 10 most dynamic objects:")
    for i in range(min(10, len(ids))):
        print(f"  Rank {i+1}: ID {ids[i]} - Score {scores[i]:.4f}")

def create_bar_chart(score_dict, output_path=None, top_n=15):
    """创建柱状图显示前N个最动态的对象
    
    Args:
        score_dict: {ID: score} 字典
        output_path: 输出图片路径
        top_n: 显示前N个对象
    """
    # 按得分从高到低排序，只取前N个
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    ids = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 创建颜色渐变
    colors = plt.cm.plasma(np.linspace(0, 1, len(ids)))
    
    # 绘制柱状图
    bars = plt.bar(range(len(ids)), scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 设置标题和标签
    plt.title(f'Top {top_n} Most Dynamic Objects', fontsize=16, fontweight='bold')
    plt.xlabel('Track ID', fontsize=12)
    plt.ylabel('Dynamic Motion Score', fontsize=12)
    
    # 设置x轴标签
    plt.xticks(range(len(ids)), [f'ID {id_}' for id_ in ids], rotation=45, ha='right')
    
    # 在柱状图上显示数值
    for i, (bar, score) in enumerate(zip(bars, scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        bar_path = output_path.replace('.png', '_bar.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        print(f"Bar chart saved to: {bar_path}")
    

if __name__ == "__main__":
    # 设置文件路径
    parser = argparse.ArgumentParser(description="图片对分割跟踪")
    parser.add_argument('--source_file', type=str, required=True, help='字典文件路径')
    args = parser.parse_args()

    score_file = os.path.join(args.source_file, "dynamic_scores.txt")
   
    output_dir = args.source_file
    
    # 加载动态得分字典
    print("Loading dynamic scores...")
    score_dict = load_dynamic_scores(score_file)
    if len(score_dict) == 0:
        print("No scores found in the file, exiting.")
        exit(1)
    scores = list(score_dict.values())
    slope_scores = []
    for score in scores:
       if score < 0.5:
            slope_scores.append(score)
    X = np.arange(len(slope_scores))
    Y = np.array(slope_scores)
    knee = KneeLocator(X, Y, curve="convex", direction="decreasing")
    knee_index = knee.knee
    if knee_index is None:
        print("No knee point found, using the last score as threshold")
        knee_index = len(slope_scores) - 1
    threshold = slope_scores[knee_index]
    print(f"Knee point found at index {knee_index} with score {threshold:.4f}")
    
    if score_dict:
        print(f"Successfully loaded {len(score_dict)} track IDs with scores")
        
        # 创建折线图
        line_plot_path = os.path.join(output_dir , "dynamic_scores_line_plot.png") 
        visualize_dynamic_scores(score_dict, threshold, str(line_plot_path))
        
        # # 创建柱状图（显示前15个最动态的对象）
        # bar_plot_path = output_dir / "dynamic_scores_bar_plot.png"
        # create_bar_chart(score_dict, str(bar_plot_path), top_n=15)
        
    else:
        print("No scores found in the file")
    
