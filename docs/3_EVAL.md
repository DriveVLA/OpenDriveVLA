# Evaluate DriveVLA

## 1. Download DriveVLA checkpoint

```shell
cd DriveVLA
mkdir checkpoints
# Download the DriveVLA checkpoint here
```

## 2. Evaluate DriveVLA

```shell
cd DriveVLA
conda activate drivevla
bash scripts/eval_drivevla.sh <CKPT_PATH> <NUM_GPU>
```

For example:

```shell
cd DriveVLA
conda activate drivevla
bash scripts/eval_drivevla.sh checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct 1
```
