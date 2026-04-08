#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import gc
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import random
import numpy as np

from clustering_head import KMeansClusteringHead

from build_contrastive_data import build_contrastive_samples
from classification_head import evaluate_embeddings_with_classification, load_text_data
from link_prediction_head import load_graph_for_link_pred, generate_embeddings, evaluate_link_prediction_with_variance

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
class PathEncoder:
    def __init__(self, model_path, max_length=512, use_gradient_checkpointing=False,
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1, lora_target_modules=None):
        self.max_length = max_length
        self.use_lora = use_lora
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        if self.use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
            self.model.enable_input_require_grads()
        else:
            self.model = base_model

        if use_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

    def encode(self, texts, batch_size=32):
        self.model.eval()
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors='pt')
                outputs = self.model(**inputs)
                batch_embeds = outputs.last_hidden_state[:, 0, :]
            embeddings.append(batch_embeds)
        if len(embeddings) > 0:
            result = torch.cat(embeddings, dim=0)
        else:
            result = torch.empty((0, self.model.config.hidden_size))
        return result


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, pos_margin=0.1, neg_margin=0.1):
        super().__init__()
        self.temperature = temperature
        self.pos_margin = pos_margin  
        self.neg_margin = neg_margin  
    
    def forward(self, query_embeds, pos_main_list, pos_hard_list, 
                neg_main_list, neg_hard_list):
        batch_size = query_embeds.size(0)
        losses = []
        
        for i in range(batch_size):
            query = query_embeds[i:i+1]
            
            if len(pos_main_list[i]) == 0:
                raise ValueError(f"Sample {i} has no positive main samples! Loss cannot be computed.")
            
            # ========== 正样本得分 pos ==========
            all_pos_scores = []
            
            # d+_1
            pos_main_tensor = torch.cat(pos_main_list[i], dim=0)  # [N, D]
            pos_main_sims = F.cosine_similarity(query, pos_main_tensor, dim=1)  # [N]
            pos_main_scores = torch.exp(pos_main_sims / self.temperature)
            all_pos_scores.append(pos_main_scores)
            pos_main_sim = pos_main_sims[0]
            
            # d+_2
            if len(pos_hard_list[i]) > 0:
                pos_hard_tensor = torch.cat(pos_hard_list[i], dim=0)
                pos_hard_sims = F.cosine_similarity(query, pos_hard_tensor, dim=1)
                # 假正屏蔽
                p_masks = (pos_hard_sims >= pos_main_sim - self.pos_margin).float()
                pos_hard_scores = p_masks * torch.exp(pos_hard_sims / self.temperature)
                all_pos_scores.append(pos_hard_scores)
            
            if all_pos_scores:
                pos_score = torch.cat(all_pos_scores, dim=0).sum()
            else:
                pos_score = query.sum() * 0.0
            
            # ========== 负样本得分 neg ==========
            all_neg_scores = []
            
            # d-_1
            if len(neg_main_list[i]) > 0:
                neg_main_tensor = torch.cat(neg_main_list[i], dim=0)
                neg_main_sims = F.cosine_similarity(query, neg_main_tensor, dim=1)
                # 假负屏蔽
                n_masks = (neg_main_sims <= pos_main_sim + self.neg_margin).float()
                neg_main_scores = n_masks * torch.exp(neg_main_sims / self.temperature)
                all_neg_scores.append(neg_main_scores)
            
            # d-_2
            if len(neg_hard_list[i]) > 0:
                neg_hard_tensor = torch.cat(neg_hard_list[i], dim=0)
                neg_hard_sims = F.cosine_similarity(query, neg_hard_tensor, dim=1)
                # 假负屏蔽
                n_masks = (neg_hard_sims <= pos_main_sim + self.neg_margin).float()
                neg_hard_scores = n_masks * torch.exp(neg_hard_sims / self.temperature)
                all_neg_scores.append(neg_hard_scores)
            
            # query_i 与其他 query_j 
            for j in range(batch_size):
                if j != i:
                    other_query = query_embeds[j:j+1]
                    sim = F.cosine_similarity(query, other_query, dim=1).squeeze()
                    # 假负屏蔽
                    mask = (sim <= pos_main_sim + self.neg_margin).float()
                    all_neg_scores.append((mask * torch.exp(sim / self.temperature)).unsqueeze(0))
            
            # d_i+ 与其他 d_j+ 
            if len(pos_main_list[i]) > 0:
                curr_pos_tensor = torch.cat(pos_main_list[i], dim=0)
                curr_pos = curr_pos_tensor[0:1]
                for j in range(batch_size):
                    if j != i and len(pos_main_list[j]) > 0:
                        other_pos_tensor = torch.cat(pos_main_list[j], dim=0)
                        other_pos = other_pos_tensor[0:1]
                        sim = F.cosine_similarity(curr_pos, other_pos, dim=1).squeeze()
                        # 假负屏蔽
                        mask = (sim <= pos_main_sim + self.neg_margin).float()
                        all_neg_scores.append((mask * torch.exp(sim / self.temperature)).unsqueeze(0))
            
            # query_i 与其他 d_j+ 
            for j in range(batch_size):
                if j != i and len(pos_main_list[j]) > 0:
                    other_pos_tensor = torch.cat(pos_main_list[j], dim=0)
                    other_pos = other_pos_tensor[0:1]
                    sim = F.cosine_similarity(query, other_pos, dim=1).squeeze()
                    # 假负屏蔽
                    mask = (sim <= pos_main_sim + self.neg_margin).float()
                    all_neg_scores.append((mask * torch.exp(sim / self.temperature)).unsqueeze(0))
            
            if all_neg_scores:
                neg_score = torch.cat(all_neg_scores, dim=0).sum()
            else:
                neg_score = query.sum() * 0.0
            
            loss = -torch.log(pos_score / (pos_score + neg_score + 1e-8))
            losses.append(loss)
        
        return torch.stack(losses).mean()


class ContrastiveDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.nl_text_pool = data['nl_text_pool']
        self.samples = data['samples']
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'query_segment': sample['query']['segment'],
            'query_nl_idx': sample['query']['nl_idx'],
            'positives': sample['positives'],
            'negatives': sample['negatives']
        }

def train_epoch(model, tokenizer, dataloader, nl_text_pool, criterion, optimizer, device, epoch, total_epochs, max_length, is_main_process=True):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", 
                       ncols=100, leave=True, disable=not is_main_process)

    for batch_idx, batch in enumerate(progress_bar, 1):
        query_texts = []            
        other_texts = []          
        other_meta = []          

        for sample_idx, sample in enumerate(batch):
            # Query
            nl_idx = sample['query_nl_idx']
            query_texts.append(nl_text_pool[nl_idx])

            # Positives
            for pos in sample['positives']:
                other_texts.append(nl_text_pool[pos['nl_idx']])
                other_meta.append(('pos', pos['type'], sample_idx))

            # Negatives
            for neg in sample['negatives']:
                other_texts.append(nl_text_pool[neg['nl_idx']])
                other_meta.append(('neg', neg['type'], sample_idx))

        batch_size = len(batch)

        max_count = len(other_texts)

        inputs_q = tokenizer(query_texts, padding=True, truncation=True, 
                             max_length=max_length, return_tensors='pt').to(device)
        outputs_q = model(**inputs_q)
        query_embeds = outputs_q.last_hidden_state[:, 0, :]

        if max_count > 0:
            with torch.inference_mode():
                chunk_size = 8
                chunks = []
                for start in range(0, len(other_texts), chunk_size):
                    end = min(start + chunk_size, len(other_texts))
                    batch_texts = other_texts[start:end]
                    batch_inputs = tokenizer(batch_texts, padding=True, truncation=True,
                                             max_length=256, return_tensors='pt').to(device)
                    batch_outputs = model(**batch_inputs, use_cache=False)
                    chunks.append(batch_outputs.last_hidden_state[:, 0, :].detach())
                other_embeds = torch.cat(chunks, dim=0)
        else:
            other_embeds = torch.empty((0, query_embeds.size(1)), device=device)

        pos_main_indices = {i: [] for i in range(batch_size)}
        pos_hard_indices = {i: [] for i in range(batch_size)}
        neg_main_indices = {i: [] for i in range(batch_size)}
        neg_hard_indices = {i: [] for i in range(batch_size)}

        for idx_o in range(len(other_texts)):
            kind, subtype, sample_idx = other_meta[idx_o]
            if kind == 'pos':
                if subtype == 'd+_1':
                    pos_main_indices[sample_idx].append(idx_o)
                else:
                    pos_hard_indices[sample_idx].append(idx_o)
            else:  # neg
                if subtype == 'd-_1':
                    neg_main_indices[sample_idx].append(idx_o)
                else:
                    neg_hard_indices[sample_idx].append(idx_o)

        def build_lists_from_other(indices_dict):
            res = []
            for i in range(batch_size):
                idxs = indices_dict.get(i, [])
                if len(idxs) > 0:
                    tensors = [other_embeds[k:k+1] for k in idxs]
                    res.append(tensors)
                else:
                    res.append([])
            return res

        pos_main_list = build_lists_from_other(pos_main_indices)
        pos_hard_list = build_lists_from_other(pos_hard_indices)
        neg_main_list = build_lists_from_other(neg_main_indices)
        neg_hard_list = build_lists_from_other(neg_hard_indices)

        loss = criterion(query_embeds, pos_main_list, pos_hard_list, neg_main_list, neg_hard_list)

        model.backward(loss)
        model.step()

        total_loss += loss.item()

        try:
            del query_texts, other_texts, other_meta
            del pos_main_list, pos_hard_list, neg_main_list, neg_hard_list
            del pos_main_indices, pos_hard_indices, neg_main_indices, neg_hard_indices
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        avg_loss = total_loss / batch_idx
        progress_bar.set_postfix({'loss': f'{total_loss/batch_idx:.4f}', 'avg_loss': f'{avg_loss:.4f}'})

    return total_loss / num_batches


def evaluate_downstream_tasks(model, tokenizer, dataset_path, device, max_length=512, batch_size=8,
                              log_details=True):

    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = (rank == 0)

    if dist.is_initialized():
        dist.barrier()

    model_to_eval = model.module if hasattr(model, 'module') else model

    results = None

    if is_main:
        model_to_eval.eval()
        texts, labels = load_text_data(dataset_path, logger=None)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long)
        texts_for_link, edge_index, num_nodes = load_graph_for_link_pred(dataset_path, logger=None)

        with torch.no_grad():
            embeddings = generate_embeddings(
                model_to_eval, tokenizer, texts_for_link,
                max_length, batch_size, device,
                logger=None, prompt=""
            )

        if log_details:
            logger.info("评估分类任务...")
        cls_results = evaluate_embeddings_with_classification(
            embeddings.cpu().numpy(),
            labels,
            test_repeat=5
        )

        if log_details:
            logger.info("评估链路预测任务...")
        link_results = evaluate_link_prediction_with_variance(
            embeddings, edge_index, num_nodes,
            test_ratio=0.2, n_runs=5, device=device,
            mlp_epochs=1000, logger=None
        )

        if log_details:
            logger.info("评估聚类任务...")
        num_clusters = int(labels.max().item() + 1)
        cluster_head = KMeansClusteringHead(
            num_clusters=num_clusters,
            use_minibatch=True
        )
        cluster_results = cluster_head.evaluate(
            embeddings=embeddings,
            labels=torch.from_numpy(labels) if not torch.is_tensor(labels) else labels
        )

        results = {
            'classification': cls_results,
            'link_prediction': link_results,
            'clustering': cluster_results
        }

        if log_details:
            logger.info(f"  分类 - Acc: {cls_results['test_acc_mean']:.2f}±{cls_results['test_acc_std']:.2f}, "
                       f"Micro-F1: {cls_results['test_micro_f1_mean']:.2f}±{cls_results['test_micro_f1_std']:.2f}, "
                       f"Macro-F1: {cls_results['test_macro_f1_mean']:.2f}±{cls_results['test_macro_f1_std']:.2f}")
            logger.info(f"  链路 - AUC: {link_results['test_auc']:.2f}±{link_results['test_auc_std']:.2f}, "
                       f"AP: {link_results['test_ap']:.2f}±{link_results['test_ap_std']:.2f}")
            logger.info(
                f"  聚类 - "
                f"NMI: {cluster_results['NMI_mean']:.4f}±{cluster_results['NMI_std']:.4f}, "
                f"ARI: {cluster_results['ARI_mean']:.4f}±{cluster_results['ARI_std']:.4f}, "
                f"F1: {cluster_results['F1_mean']:.4f}±{cluster_results['F1_std']:.4f}, "
                f"ACC: {cluster_results['ACC_mean']:.4f}±{cluster_results['ACC_std']:.4f}"
            )

    if dist.is_initialized():
        obj = [results] if is_main else [None]
        dist.broadcast_object_list(obj, src=0)
        results = obj[0]
        dist.barrier()

    model.train()

    return results


def _log_eval_metrics_from_results(epoch_results):
    """仅打印下游指标（与 evaluate_downstream_tasks 中 log_details 块一致）。"""
    cls_results = epoch_results['classification']
    link_results = epoch_results['link_prediction']
    cluster_results = epoch_results['clustering']
    logger.info(f"  分类 - Acc: {cls_results['test_acc_mean']:.2f}±{cls_results['test_acc_std']:.2f}, "
               f"Micro-F1: {cls_results['test_micro_f1_mean']:.2f}±{cls_results['test_micro_f1_std']:.2f}, "
               f"Macro-F1: {cls_results['test_macro_f1_mean']:.2f}±{cls_results['test_macro_f1_std']:.2f}")
    logger.info(f"  链路 - AUC: {link_results['test_auc']:.2f}±{link_results['test_auc_std']:.2f}, "
               f"AP: {link_results['test_ap']:.2f}±{link_results['test_ap_std']:.2f}")
    logger.info(
        f"  聚类 - "
        f"NMI: {cluster_results['NMI_mean']:.4f}±{cluster_results['NMI_std']:.4f}, "
        f"ARI: {cluster_results['ARI_mean']:.4f}±{cluster_results['ARI_std']:.4f}, "
        f"F1: {cluster_results['F1_mean']:.4f}±{cluster_results['F1_std']:.4f}, "
        f"ACC: {cluster_results['ACC_mean']:.4f}±{cluster_results['ACC_std']:.4f}"
    )


def _append_to_results_file(results_file, training_results, config_tag, is_main_process):
    if not is_main_process:
        return

    if config_tag:
        training_results['config_tag'] = config_tag

    all_results = []
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = []
        except (json.JSONDecodeError, ValueError):
            all_results = []

    # ===== 按 config_tag 覆盖，而不是重复 append =====
    updated = False
    for i, item in enumerate(all_results):
        if item.get('config_tag') == config_tag:
            all_results[i] = training_results
            updated = True
            break

    if not updated:
        all_results.append(training_results)

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    logger.info(f"结果已写入: {results_file} (配置: {config_tag})")


def prepare_contrastive_samples(args):
    dataset_path = f'./datasets/{args.dataset}/{args.dataset}.pt'
    data_dict = torch.load(dataset_path, weights_only=False)
    labels = data_dict['y']
    num_classes = labels.max().item() + 1
    num_anchors = args.anchors * num_classes
    
    high_split = f'./datasets/{args.dataset}/{args.dataset}_split_high_diversity_paths_{num_anchors}_{args.cuttoff}.json'
    low_split = f'./datasets/{args.dataset}/{args.dataset}_split_low_diversity_paths_{num_anchors}_{args.cuttoff}.json'
    filtered_low = f'./datasets/{args.dataset}/{args.dataset}_filtered_low_diversity_paths_{num_anchors}_{args.cuttoff}.json'
    high_nl = f'./datasets/{args.dataset}/{args.dataset}_high_nl_paths_{num_anchors}_{args.cuttoff}.json'
    low_nl = f'./datasets/{args.dataset}/{args.dataset}_low_nl_paths_{num_anchors}_{args.cuttoff}.json'
    samples_file = f'./datasets/{args.dataset}/{args.dataset}_contrastive_samples_{num_anchors}_{args.cuttoff}.json'
    
    if os.path.exists(samples_file):
        logger.info(f"使用已存在的对比学习样本文件: {samples_file}")
    else:
        logger.info("构建对比学习样本...")
        build_contrastive_samples(
            high_split_path=high_split,
            low_split_path=low_split,
            high_nl_path=high_nl,
            low_nl_path=low_nl,
            filtered_low_path=filtered_low,
            output_path=samples_file
        )
        logger.info(f"样本构建完成: {samples_file}")
    
    return samples_file


def main():
    parser = argparse.ArgumentParser(description='对比学习训练')
    parser.add_argument('--dataset', type=str, default='cora', help='数据集名称')
    parser.add_argument('--anchors', type=int, default=20, help='每个类的anchor数量')
    parser.add_argument('--cuttoff', type=float, default=0.02, help='路径选择的cutoff阈值')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--temperature', type=float, default=0.07, help='温度参数')
    parser.add_argument('--pos_margin', type=float, default=0.1, help='假正屏蔽边界')
    parser.add_argument('--neg_margin', type=float, default=0.1, help='假负屏蔽边界')
    parser.add_argument('--model_path', type=str, default='/data/home/xmju/models/Qwen3-Embedding-0.6B', help='模型路径')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='输出目录')
    parser.add_argument('--eval_epochs', type=int, default=1, help='每隔多少轮评估一次下游任务')
    parser.add_argument('--skip_baseline_eval', action='store_true', help='跳过训练前的baseline评估')
    parser.add_argument('--max_length', type=int, default=512, help='文本最大长度')
    parser.add_argument("--deepspeed", type=str, default="ds_config.json", help="DeepSpeed config file path")
    parser.add_argument('--use_lora', action='store_true', help='使用LoRA微调')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=None, 
                       help='LoRA目标模块')
    parser.add_argument('--local_rank', type=int, default=-1, help='本地GPU rank（')
    parser.add_argument('--results_file', type=str, default=None, help='统一的结果文件路径')
    parser.add_argument('--config_tag', type=str, default=None, help='配置标签')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于保证实验可复现性')

    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    samples_file = prepare_contrastive_samples(args)
    with open(samples_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    nl_text_pool = data['nl_text_pool']

    dataset = ContrastiveDataset(samples_file)
    sampler = DistributedSampler(dataset) if torch.distributed.is_initialized() else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda x: x
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    max_length = getattr(args, 'max_length', 512)
    encoder = PathEncoder(
        args.model_path,
        max_length=max_length,
        use_gradient_checkpointing=False,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules
    )
    model = encoder.model

    criterion = ContrastiveLoss(temperature=args.temperature, 
                               pos_margin=args.pos_margin, 
                               neg_margin=args.neg_margin)

    model_params = model.parameters()
    if args.use_lora:
        trainable_params = [p for p in model_params if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model_params, lr=args.lr)

    import json as _json
    with open(args.deepspeed) as f:
        ds_config = _json.load(f)
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config_params=ds_config
    )

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_main_process = (rank == 0)
    else:
        rank = 0
        world_size = 1
        is_main_process = True

    logger.info(f"[Rank {rank}] using cuda:{torch.cuda.current_device()}")

    if is_main_process:
        logger.info(f"开始训练 - 数据集: {args.dataset}")
        logger.info(f"加载模型: {args.model_path}")

    training_results = {
        'config': {
            'dataset': args.dataset,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'temperature': args.temperature,
            'seed': args.seed
        },
        'best_eval': None
    }
    best_cls_acc = float('-inf')
    best_epoch = None

    def _best_eval_ref_for_log():
        if best_epoch is None:
            return "无"
        if best_epoch == 0:
            return "baseline"
        return f"epoch {best_epoch}"

    dataset_path = f'./datasets/{args.dataset}/{args.dataset}.pt'
    dataset_dir = f'./datasets/{args.dataset}'
    
    if not args.config_tag:
        lora_suffix = f"-lora{args.lora_r}" if args.use_lora else ""
        args.config_tag = f"{args.dataset}-temp{args.temperature}-lr{args.lr}-bs{args.batch_size}{lora_suffix}"
    
    if not args.results_file:
        args.results_file = f"{dataset_dir}/contrastive_training_results.json"

    args.output_dir = f"{dataset_dir}/checkpoints/{args.config_tag}"

    if dist.is_initialized():
        dist.barrier()
    if not args.skip_baseline_eval:
        if is_main_process:
            logger.info("训练前评估（Baseline）")
        baseline_results = evaluate_downstream_tasks(
            model, tokenizer, dataset_path, 
            args.device, max_length=max_length, batch_size=8 
        )
        if is_main_process:
            logger.info("="*70)
        if dist.is_initialized():
            dist.barrier()
    else:
        baseline_results = None

    if baseline_results and 'classification' in baseline_results:
        best_cls_acc = baseline_results['classification']['test_acc_mean']
        best_epoch = 0

    if (
        is_main_process
        and baseline_results
        and 'classification' in baseline_results
        and 'link_prediction' in baseline_results
        and 'clustering' in baseline_results
    ):
        baseline_summary = {
            'classification': {
                'acc': f"{baseline_results['classification']['test_acc_mean']:.2f}±{baseline_results['classification']['test_acc_std']:.2f}",
                'micro_f1': f"{baseline_results['classification']['test_micro_f1_mean']:.2f}±{baseline_results['classification']['test_micro_f1_std']:.2f}",
                'macro_f1': f"{baseline_results['classification']['test_macro_f1_mean']:.2f}±{baseline_results['classification']['test_macro_f1_std']:.2f}"
            },
            'link_prediction': {
                'auc': f"{baseline_results['link_prediction']['test_auc']:.2f}±{baseline_results['link_prediction']['test_auc_std']:.2f}",
                'ap': f"{baseline_results['link_prediction']['test_ap']:.2f}±{baseline_results['link_prediction']['test_ap_std']:.2f}"
            },
            'clustering': {
                'nmi': f"{baseline_results['clustering']['NMI_mean']:.4f}±{baseline_results['clustering']['NMI_std']:.4f}",
                'ari': f"{baseline_results['clustering']['ARI_mean']:.4f}±{baseline_results['clustering']['ARI_std']:.4f}",
                'f1': f"{baseline_results['clustering']['F1_mean']:.4f}±{baseline_results['clustering']['F1_std']:.4f}",
                'acc': f"{baseline_results['clustering']['ACC_mean']:.4f}±{baseline_results['clustering']['ACC_std']:.4f}"
            }
        }
        training_results['baseline'] = baseline_summary
        training_results['best_eval'] = {
            'epoch': 0,
            'source': 'baseline',
            'classification': baseline_summary['classification'],
            'link_prediction': baseline_summary['link_prediction'],
            'clustering': baseline_summary['clustering'],
        }
        _append_to_results_file(args.results_file, training_results, args.config_tag, is_main_process)

    if is_main_process:
        logger.info("="*70)
        logger.info("开始训练...")
        os.makedirs(args.output_dir, exist_ok=True)

    if dist.is_initialized():
        dist.barrier()

    for epoch in range(1, args.epochs + 1):
        if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        avg_loss = train_epoch(model, tokenizer, dataloader, nl_text_pool, 
                              criterion, optimizer, args.device, epoch, args.epochs, max_length, is_main_process)

        if is_main_process:
            logger.info(f"Epoch {epoch}/{args.epochs} - 平均损失: {avg_loss:.4f}")

        if dist.is_initialized():
            dist.barrier()

        need_eval = (epoch % args.eval_epochs == 0) or (epoch == args.epochs)

        if need_eval:
            if is_main_process:
                logger.info("-" * 70)
                logger.info(f"Epoch {epoch} 评估开始")

            epoch_results = evaluate_downstream_tasks(
                model, tokenizer, dataset_path,
                args.device, max_length=max_length, batch_size=8,
                log_details=False
            )

            if epoch_results:
                cur_acc = epoch_results['classification']['test_acc_mean']
                improved = cur_acc > best_cls_acc
                if improved:
                    best_cls_acc = cur_acc
                    best_epoch = epoch
                    if is_main_process:
                        logger.info(f"Epoch {epoch} 刷新最佳分类 Acc={cur_acc:.2f}，详细指标：")
                        _log_eval_metrics_from_results(epoch_results)
                        epoch_summary = {
                            'epoch': epoch,
                            'source': 'training',
                            'classification': {
                                'acc': f"{epoch_results['classification']['test_acc_mean']:.2f}±{epoch_results['classification']['test_acc_std']:.2f}",
                                'micro_f1': f"{epoch_results['classification']['test_micro_f1_mean']:.2f}±{epoch_results['classification']['test_micro_f1_std']:.2f}",
                                'macro_f1': f"{epoch_results['classification']['test_macro_f1_mean']:.2f}±{epoch_results['classification']['test_macro_f1_std']:.2f}"
                            },
                            'link_prediction': {
                                'auc': f"{epoch_results['link_prediction']['test_auc']:.2f}±{epoch_results['link_prediction']['test_auc_std']:.2f}",
                                'ap': f"{epoch_results['link_prediction']['test_ap']:.2f}±{epoch_results['link_prediction']['test_ap_std']:.2f}"
                            },
                            'clustering': {
                                'nmi': f"{epoch_results['clustering']['NMI_mean']:.4f}±{epoch_results['clustering']['NMI_std']:.4f}",
                                'ari': f"{epoch_results['clustering']['ARI_mean']:.4f}±{epoch_results['clustering']['ARI_std']:.4f}",
                                'f1': f"{epoch_results['clustering']['F1_mean']:.4f}±{epoch_results['clustering']['F1_std']:.4f}",
                                'acc': f"{epoch_results['clustering']['ACC_mean']:.4f}±{epoch_results['clustering']['ACC_std']:.4f}"
                            }
                        }
                        training_results['best_eval'] = epoch_summary
                        logger.info("指标与 checkpoint 已按最佳分类 Acc 更新")
                        _append_to_results_file(args.results_file, training_results, args.config_tag, is_main_process)
                    if dist.is_initialized():
                        dist.barrier()
                    model.save_checkpoint(args.output_dir, tag="best_acc")
                elif is_main_process:
                    logger.info(
                        f"Epoch {epoch} 评估完成 | Acc={cur_acc:.2f}（当前最佳 Acc={best_cls_acc:.2f} @ {_best_eval_ref_for_log()}），不写入指标文件"
                    )

            if dist.is_initialized():
                dist.barrier()

    if dist.is_initialized():
        dist.barrier()
    
    if is_main_process:
        _append_to_results_file(args.results_file, training_results, args.config_tag, is_main_process)
        be = training_results.get('best_eval') or {}
        if be.get('source') == 'baseline':
            logger.info(
                "全程未出现高于 Baseline 的分类 Acc；best_eval 与结果文件中的最佳指标保留为训练前 Baseline。"
            )
        logger.info("="*70)
        logger.info(f"训练完成！")


if __name__ == "__main__":
    main()

