import os
import json
import torch
import argparse

import numpy as np
import networkx as nx
import torch.nn.functional as F

import torch.multiprocessing as mp

from path_utils import (
    set_everything,
    generate_features_with_llm,
    diversity_score,
    plot_diversity_label_consistency
)
from select_anchors import kmeans


def prepare_paths(args):
    """
    第一阶段：从加载数据集到按比例筛选高低多样性路径
    """
    assert args.gpu_id in range(0, 8)
    # torch.cuda.set_device(args.gpu_id)
    set_everything(args.seed)

    ### 加载数据集
    dataset_path = f'./datasets/{args.dataset}/{args.dataset}.pt'
    data_dict = torch.load(dataset_path, weights_only=False)
    raw_texts, labels, edge_index  = data_dict['raw_texts'], data_dict['y'], data_dict['edge_index']

    num_classes = labels.max().item()+1
    args.num_anchors = args.anchors * num_classes
    num_nodes = len(raw_texts)
    
    ### 加载数据集特征
    feature_path = f'./datasets/{args.dataset}/{args.dataset}_features.pt'
    if os.path.exists(feature_path):
        features = torch.load(feature_path)
    else:
        features = generate_features_with_llm(args, raw_texts)
        torch.save(features, feature_path)
        
    ### 加载锚点集
    anchors_path = f'./datasets/{args.dataset}/{args.dataset}_anchors_{args.num_anchors}.pt'
    anchors_json_path = f'./datasets/{args.dataset}/{args.dataset}_anchors_{args.num_anchors}.json'
    if os.path.exists(anchors_path):
        anchors = torch.load(anchors_path)
    else:
        predict_labels, centers = kmeans(X=features, num_clusters=args.num_anchors, distance="euclidean", device="cuda")  
        distances = torch.cdist(features, centers, p=2)
        anchors = torch.argmin(distances, dim=0).cpu().tolist()
        torch.save(anchors, anchors_path)
        
    if not os.path.exists(anchors_json_path):
        with open(anchors_json_path, 'w', encoding='utf-8') as f:
            json.dump(anchors, f, indent=4)
        
    ### 加载最短路径   
    paths_path = f'./datasets/{args.dataset}/{args.dataset}_shortest_paths_anchors_{args.num_anchors}.pt'
    paths_json_path = f'./datasets/{args.dataset}/{args.dataset}_shortest_paths_anchors_{args.num_anchors}.json'
    if os.path.exists(paths_path):
        all_paths = torch.load(paths_path)
    else:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_index.t().tolist())
        all_paths = []
        for anchor in anchors:
            all_shortest_paths = nx.single_source_shortest_path(G, anchor)
            for node, path in all_shortest_paths.items():
                if len(path) - 1 >= 2:
                    all_paths.append(path)          
        torch.save(all_paths, paths_path)
        
    if not os.path.exists(paths_json_path):
        with open(paths_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_paths, f, indent=4)

    ### 计算路径多样性分数并保存路径详细信息（节点、标签、多样性分数）
    scores_path = f'./datasets/{args.dataset}/{args.dataset}_scores_anchors_{args.num_anchors}.pt'   
    scores_json_path = f'./datasets/{args.dataset}/{args.dataset}_scores_anchors_{args.num_anchors}.json'
    if os.path.exists(scores_path):
        scores = torch.load(scores_path)
    else: 
        scores = []
        for path in all_paths:
            score = diversity_score(path, features)
            path_labels = labels[path].cpu().tolist()
            scores.append({
                'path': path,
                'labels': path_labels,
                'diversity_score': score
            })
        torch.save(scores, scores_path)

    if not os.path.exists(scores_json_path):
        scores_for_json = []
        for item in scores:
            scores_for_json.append({
                'path': item['path'],
                'labels': item['labels'],
                'diversity_score': item['diversity_score']
            })
        with open(scores_json_path, 'w', encoding='utf-8') as f:
            json.dump(scores_for_json, f, indent=4)
            
    ### 按比例筛选高/低多样性路径
    diversity_scores = [item['diversity_score'] for item in scores]

    scores_array = np.array(diversity_scores)
    low_percentile = np.percentile(scores_array, args.cuttoff*100)
    high_percentile = np.percentile(scores_array, (1 - args.cuttoff)*100)

    high_diversity_paths = [item for item in scores if item['diversity_score'] >= high_percentile]
    low_diversity_paths = [item for item in scores if item['diversity_score'] <= low_percentile]
    high_diversity_paths.sort(key=lambda x: x['diversity_score'], reverse=True)
    low_diversity_paths.sort(key=lambda x: x['diversity_score'], reverse=True)
    print(f"Number of high diversity paths (top {args.cuttoff*100}%): {len(high_diversity_paths)}")
    print(f"Number of low diversity paths (bottom {args.cuttoff*100}%): {len(low_diversity_paths)}")

    high_paths_path = f'./datasets/{args.dataset}/{args.dataset}_high_diversity_paths_{args.num_anchors}_{args.cuttoff}.json'
    low_paths_path = f'./datasets/{args.dataset}/{args.dataset}_low_diversity_paths_{args.num_anchors}_{args.cuttoff}.json'

    with open(high_paths_path, 'w', encoding='utf-8') as f:
        json.dump(high_diversity_paths, f, indent=4)
    with open(low_paths_path, 'w', encoding='utf-8') as f:
        json.dump(low_diversity_paths, f, indent=4)
    
    print(f"第一阶段完成：路径筛选结果已保存到 {high_paths_path} 和 {low_paths_path}")


def main(args):
    prepare_paths(args)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) 
    parser = argparse.ArgumentParser(description='PathContrast Configuration')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--anchors', type=int, default=20, help='anchors in each class')
    parser.add_argument('--cuttoff', type=float, default=0.02, help='cutoff for path selection')
    parser.add_argument('--embedding_model_name', type=str, default='/data/home/xmju/models/Qwen3-Embedding-0.6B')
    args = parser.parse_args()
    args.prompt = args.embedding_model_name.split("/")[-1]
    
    main(args)
