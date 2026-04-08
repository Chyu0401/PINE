## Environment setup

```bash
pip install -r requirements.txt
```

## Step 1: Anchor selection

Run `main_stage1.py`.

## Step 2: Path preparation

Run `main_stage2.py`.

## Step 3: Positive/negative sample construction and training

Run via bash:

```bash
bash train_params.sh
```

This script invokes `contrastive_train.py` for training.
