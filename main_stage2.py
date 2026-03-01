import os
import json
import torch
import argparse
import logging

import torch.multiprocessing as mp

from path_utils import (
    low_diversity_filter,
    generate_edge_explanations,
    split_high_diversity_paths,
    split_low_diversity_paths,
    build_natural_language_paths
)

# 配置 logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def process_with_llm(args):
    logger.info(f"CUDA initialized before loading data: {torch.cuda.is_initialized()}")
    
    ### 加载数据集（需要 raw_texts 和 edge_index）
    dataset_path = f'./datasets/{args.dataset}/{args.dataset}.pt'
    data_dict = torch.load(dataset_path, weights_only=False)
    raw_texts, labels, edge_index = data_dict['raw_texts'], data_dict['y'], data_dict['edge_index']
    
    num_classes = labels.max().item()+1
    args.num_anchors = args.anchors * num_classes
    
    ### 加载高低多样性路径
    high_paths_path = f'./datasets/{args.dataset}/{args.dataset}_high_diversity_paths_{args.num_anchors}_{args.cuttoff}.json'
    low_paths_path = f'./datasets/{args.dataset}/{args.dataset}_low_diversity_paths_{args.num_anchors}_{args.cuttoff}.json'
    
    if not os.path.exists(high_paths_path) or not os.path.exists(low_paths_path):
        raise FileNotFoundError(f"请先运行第一阶段 prepare_paths 生成路径文件：{high_paths_path} 和 {low_paths_path}")
    
    with open(high_paths_path, 'r', encoding='utf-8') as f:
        high_diversity_paths = json.load(f)
    with open(low_paths_path, 'r', encoding='utf-8') as f:
        low_diversity_paths = json.load(f)
    
    logger.info(f"加载路径：高多样性 {len(high_diversity_paths)} 条，低多样性 {len(low_diversity_paths)} 条")

    ### 绘制多样性分数分布图
    # plot_diversity_label_consistency(
    #     scores=scores,
    #     dataset_name=args.dataset,
    #     save_path=f'./datasets/{args.dataset}/{args.dataset}_diversity_vs_label_consistency_{args.num_anchors}.png'
    # )

    ### 使用大模型过滤低多样性路径
    filtered_low_paths_path = f'./datasets/{args.dataset}/{args.dataset}_filtered_low_diversity_paths_{args.num_anchors}_{args.cuttoff}.json'

    if os.path.exists(filtered_low_paths_path):
        with open(filtered_low_paths_path, 'r', encoding='utf-8') as f:
            filtered_low_diversity_paths = json.load(f)
    else:
        filtered_low_diversity_paths = low_diversity_filter(args.instruct_model_name, low_diversity_paths, raw_texts, args)

        os.makedirs(os.path.dirname(filtered_low_paths_path), exist_ok=True)
        with open(filtered_low_paths_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_low_diversity_paths, f, indent=4)

    true_count = sum(1 for item in filtered_low_diversity_paths
                     if item.get('llm_filter_result', {}).get('answer', 'false') == 'true')
    false_count = len(filtered_low_diversity_paths) - true_count
    logger.info("LLM filtering completed.")
    logger.info(f"Paths with same research topic: {true_count}")
    logger.info(f"Paths with different research topics: {false_count}")

    ### 生成边语义
    edge_explanations_path = f'./datasets/{args.dataset}/{args.dataset}_edge_explanations.json'

    if os.path.exists(edge_explanations_path):
        with open(edge_explanations_path, 'r', encoding='utf-8') as f:
            all_explanations = json.load(f)
    else:
        all_explanations = generate_edge_explanations(edge_index, raw_texts, args.instruct_model_name, args.dataset)

        os.makedirs(os.path.dirname(edge_explanations_path), exist_ok=True)
        with open(edge_explanations_path, 'w', encoding='utf-8') as f:
            json.dump(all_explanations, f, indent=4, ensure_ascii=False)

    ### 高多样性路径划分
    split_high_paths_path = f'./datasets/{args.dataset}/{args.dataset}_split_high_diversity_paths_{args.num_anchors}_{args.cuttoff}.json'

    if os.path.exists(split_high_paths_path):
        with open(split_high_paths_path, 'r', encoding='utf-8') as f:
            split_results = json.load(f)
    else:
        split_results = split_high_diversity_paths(args.instruct_model_name, high_diversity_paths, raw_texts, args.dataset)

        os.makedirs(os.path.dirname(split_high_paths_path), exist_ok=True)
        with open(split_high_paths_path, 'w', encoding='utf-8') as f:
            json.dump(split_results, f, indent=4)

    ### 低多样性路径划分
    split_low_paths_path = f'./datasets/{args.dataset}/{args.dataset}_split_low_diversity_paths_{args.num_anchors}_{args.cuttoff}.json'

    if os.path.exists(split_low_paths_path):
        with open(split_low_paths_path, 'r', encoding='utf-8') as f:
            split_low_results = json.load(f)
    else:
        split_low_results = split_low_diversity_paths(filtered_low_diversity_paths)

        os.makedirs(os.path.dirname(split_low_paths_path), exist_ok=True)
        with open(split_low_paths_path, 'w', encoding='utf-8') as f:
            json.dump(split_low_results, f, indent=4)

    ### 构建自然语言路径
    high_nl_paths_path = f'./datasets/{args.dataset}/{args.dataset}_high_nl_paths_{args.num_anchors}_{args.cuttoff}.json'
    low_nl_paths_path = f'./datasets/{args.dataset}/{args.dataset}_low_nl_paths_{args.num_anchors}_{args.cuttoff}.json'
    
    nl_paths_result = build_natural_language_paths(
        high_diversity_split_results=split_results,
        low_diversity_split_results=split_low_results,
        filtered_low_diversity_paths=filtered_low_diversity_paths,
        edge_explanations=all_explanations,
        raw_texts=raw_texts,
        high_output_path=high_nl_paths_path,
        low_output_path=low_nl_paths_path,
        dataset_name=args.dataset
    )
    
    logger.info("第二阶段完成：所有结果已保存")

    breakpoint()


def main(args):
    process_with_llm(args)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True) 
    parser = argparse.ArgumentParser(description='PathContrast - LLM Processing Stage')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--anchors', type=int, default=20, help='anchors in each class')
    parser.add_argument('--cuttoff', type=float, default=0.02, help='cutoff for path selection')
    parser.add_argument('--instruct_model_name', type=str, default='/data/home/xmju/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554')
    args = parser.parse_args()
    
    main(args)
