import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
from pathlib import Path
project_root = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import time
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.optim import Adam
import functools


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        pass

    def __call__(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict) -> dict:
        for key in ['train', 'test', 'valid']:
            assert key in split

        result = self.evaluate(x, y, split)
        return result


class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


class LREvaluator(BaseEvaluator):
    def __init__(self, num_epochs: int = 5000, learning_rate: float = 0.01,
                 weight_decay: float = 0.0, test_interval: int = 10):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.test_interval = test_interval

    def evaluate(self, x: torch.FloatTensor, y: torch.LongTensor, split: dict):
        device = x.device
        x = x.detach().to(device)
        input_dim = x.size()[1]
        y = y.to(device)
        num_classes = y.max().item() + 1
        classifier = LogisticRegression(input_dim, num_classes).to(device)
        optimizer = Adam(classifier.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        output_fn = nn.LogSoftmax(dim=-1)
        criterion = nn.NLLLoss()

        best_test_acc = 0

        with tqdm(total=self.num_epochs, desc='(LR)',
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]') as pbar:
            for epoch in range(self.num_epochs):
                classifier.train()
                optimizer.zero_grad()

                output = classifier(x[split['train']])
                loss = criterion(output_fn(output), y[split['train']])

                loss.backward()
                optimizer.step()

                if (epoch + 1) % self.test_interval == 0:
                    classifier.eval()
                    y_test = y[split['test']].detach().cpu().numpy()
                    y_test_pred = classifier(x[split['test']]).argmax(-1).detach().cpu().numpy()
                    test_acc = (y_test == y_test_pred).mean()

                    if test_acc > best_test_acc:
                        best_test_acc = test_acc

                    pbar.set_postfix({'best_test_acc': best_test_acc})
                    pbar.update(self.test_interval)

        return {
            'best_test_acc': best_test_acc
        }


def get_split(num_samples: int, train_ratio: float = 0.8, test_ratio: float = 0.2):
    assert train_ratio + test_ratio == 1.0
    train_size = int(num_samples * train_ratio)
    test_size = int(num_samples * test_ratio)
    indices = torch.randperm(num_samples)
    return {
        'train': indices[:train_size],
        'test': indices[train_size:]
    }


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def print_statistics(statistics, function_name):
    metrics = []
    for key in statistics.keys():
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        metrics.append(f'{key}={mean:.2f}±{std:.2f}')
    print(f'{function_name}: {", ".join(metrics)}')


@dataclass
class ClassificationArguments:
    model_path: str = field(default=None)  # 微调后的模型路径
    dataset_path: str = field(default="datasets/citeseer.pt")
    output_dir: str = field(default="classification_results")
    batch_size: int = field(default=32)
    num_epochs: int = field(default=10)
    learning_rate: float = field(default=1e-4)
    max_length: int = field(default=720)
    test_size: float = field(default=0.2)
    random_state: int = field(default=42)
    device_id: int = field(default=0)


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 720, prompt: str = "Description: "):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt = prompt
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 添加描述前缀（可配置prompt）
        full_text = self.prompt + text
        
        # 分词
        tokens = self.tokenizer(
            full_text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class ClassificationHead(nn.Module):
    
    def __init__(self, embedding_model, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.embedding_model = embedding_model
        self.num_classes = num_classes
        
        # 冻结embedding模型的参数
        for param in self.embedding_model.parameters():
            param.requires_grad = False
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embedding_model.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
    
    def forward(self, input_ids, attention_mask):
        # 获取embedding
        with torch.no_grad():
            outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            embeddings = self.last_token_pool(last_hidden, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 分类
        logits = self.classifier(embeddings)
        return logits


class ClassificationTrainer:
    
    def __init__(self, model, tokenizer, args, device, logger):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.args = args
        self.device = device
        self.logger = logger
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=3,
            gamma=0.8
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            logits = self.model(input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测结果
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        self.train_losses.append(avg_loss)
        return avg_loss, accuracy
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self, train_dataloader, val_dataloader, num_epochs):
        """训练模型"""
        best_accuracy = 0
        best_epoch = 0
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(train_dataloader)
            
            # 验证
            val_results = self.evaluate(val_dataloader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存最佳模型
            if val_results['accuracy'] > best_accuracy:
                best_accuracy = val_results['accuracy']
                best_epoch = epoch + 1
                self.save_model("best_classification_model")
            
            # 每10个epoch或最后一个epoch输出进度
            if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}: 验证准确率 {val_results['accuracy']:.4f}")
        
        return best_accuracy
    
    def save_model(self, model_name):
        """保存模型"""
        model_path = os.path.join(self.args.output_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        torch.save(self.model.state_dict(), os.path.join(model_path, 'classification_head.pt'))
        self.tokenizer.save_pretrained(model_path)
        
        # 保存训练历史
        training_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }
        torch.save(training_history, os.path.join(model_path, 'training_history.pt'))
        
        self.logger.info(f"模型已保存到: {model_path}")


def load_text_data(dataset_path: str, logger) -> Tuple[List[str], List[int]]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    
    data = torch.load(dataset_path, weights_only=False)
    
    # 提取文本和标签
    raw_texts = data['raw_texts']
    labels = data['y']
    
    # 处理raw_texts格式
    if isinstance(raw_texts, list):
        texts = raw_texts
    elif isinstance(raw_texts, dict):
        # 按节点ID排序
        sorted_items = sorted(raw_texts.items())
        texts = [item[1] for item in sorted_items]
    else:
        raise ValueError(f"不支持的raw_texts格式: {type(raw_texts)}")
    
    # 处理标签格式
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # 确保文本和标签数量一致
    min_len = min(len(texts), len(labels))
    texts = texts[:min_len]
    labels = labels[:min_len]
    
    # 统计标签分布
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    if logger:
        logger.info(f"数据集: {len(texts)} 个样本, {len(unique_labels)} 个类别")
    
    return texts, labels.tolist()


def label_classification(embeddings, data, dataset_name, test_repeat=5):
    y = data.y
    test = torch.zeros(test_repeat)
    val = torch.zeros(test_repeat)
    for num in range(test_repeat):  
        if dataset_name == 'arxiv':
            split = {
                'train': data.split_idx['train'].cpu(),
                'valid': data.split_idx['valid'].cpu(),
                'test': data.split_idx['test'].cpu()}
        else:
            split = get_split(embeddings.shape[0], train_ratio=0.1, test_ratio=0.8)
        logreg = LREvaluator(num_epochs=10000)
        result = logreg.evaluate(embeddings, y, split)
        test[num] = result['best_test_acc']
        val[num] = result['best_val_acc']
    return test.mean().item()*100, test.std().item()*100, val.mean().item()*100, val.std().item()*100


def evaluate_embeddings_with_classification(embeddings, labels, test_repeat=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    
    embeddings = torch.FloatTensor(embeddings)
    labels = torch.LongTensor(labels)
    
    test_accs = []
    test_micro_f1s = []
    test_macro_f1s = []
    train_accs = []
    train_micro_f1s = []
    train_macro_f1s = []
    
    for _ in range(test_repeat):
        split = get_split(embeddings.shape[0], train_ratio=0.8, test_ratio=0.2)
        
        # 使用sklearn的LogisticRegression进行快速训练
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(embeddings[split['train']], labels[split['train']])
        
        # 测试集预测
        test_pred = clf.predict(embeddings[split['test']])
        test_true = labels[split['test']]
        test_acc = accuracy_score(test_true, test_pred)
        test_micro_f1 = f1_score(test_true, test_pred, average='micro')
        test_macro_f1 = f1_score(test_true, test_pred, average='macro')
        
        # 训练集预测
        train_pred = clf.predict(embeddings[split['train']])
        train_true = labels[split['train']]
        train_acc = accuracy_score(train_true, train_pred)
        train_micro_f1 = f1_score(train_true, train_pred, average='micro')
        train_macro_f1 = f1_score(train_true, train_pred, average='macro')
        
        test_accs.append(test_acc)
        test_micro_f1s.append(test_micro_f1)
        test_macro_f1s.append(test_macro_f1)
        train_accs.append(train_acc)
        train_micro_f1s.append(train_micro_f1)
        train_macro_f1s.append(train_macro_f1)
    
    return {
        'test_acc_mean': np.mean(test_accs) * 100,
        'test_acc_std': np.std(test_accs) * 100,
        'test_micro_f1_mean': np.mean(test_micro_f1s) * 100,
        'test_micro_f1_std': np.std(test_micro_f1s) * 100,
        'test_macro_f1_mean': np.mean(test_macro_f1s) * 100,
        'test_macro_f1_std': np.std(test_macro_f1s) * 100,
        'train_acc_mean': np.mean(train_accs) * 100,
        'train_acc_std': np.std(train_accs) * 100,
        'train_micro_f1_mean': np.mean(train_micro_f1s) * 100,
        'train_micro_f1_std': np.std(train_micro_f1s) * 100,
        'train_macro_f1_mean': np.mean(train_macro_f1s) * 100,
        'train_macro_f1_std': np.std(train_macro_f1s) * 100
    }


def main():
    args = ClassificationArguments()
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    texts, labels = load_text_data(args.dataset_path, logger)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side='left',
        local_files_only=True
    )
    
    embedding_model = AutoModel.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.float32
    )

    embedding_model = embedding_model.to(device)
 
    num_classes = len(set(labels))
    embeddings = []

    dataset = TextDataset(texts, labels, tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    embedding_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = embedding_model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            
            # 使用last token pooling
            left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
            if left_padding:
                emb = last_hidden[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden.shape[0]
                batch_indices = torch.arange(batch_size, device=last_hidden.device)
                emb = last_hidden[batch_indices, sequence_lengths]
            
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu())
    
    embeddings = torch.cat(embeddings, dim=0)
    labels_tensor = torch.LongTensor(labels)

    results = evaluate_embeddings_with_classification(embeddings, labels_tensor, test_repeat=10)
    
    logger.info(f"分类评估完成: 测试准确率 {results['test_acc_mean']:.2f}±{results['test_acc_std']:.2f}%")

    final_results = {
        'test_acc_mean': results['test_acc_mean'],
        'test_acc_std': results['test_acc_std'],
        'test_micro_f1_mean': results['test_micro_f1_mean'],
        'test_micro_f1_std': results['test_micro_f1_std'],
        'test_macro_f1_mean': results['test_macro_f1_mean'],
        'test_macro_f1_std': results['test_macro_f1_std'],
        'num_classes': num_classes,
        'total_samples': len(texts),
        'embedding_dim': embeddings.shape[1]
    }
    
    results_path = os.path.join(args.output_dir, 'classification_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
