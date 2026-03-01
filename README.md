## 环境准备

```bash
pip install -r requirements.txt
```

## 第一步：锚点选择

运行 `main_stage1.py`。

## 第二步：路径准备

运行 `main_stage2.py`。

## 第三步：正负样本构建和训练

通过 bash 运行：

```bash
bash train_params.sh
```

该脚本会调用 `contrastive_train.py` 进行训练。
