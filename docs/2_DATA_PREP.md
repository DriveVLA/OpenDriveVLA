# Prepare data for DriveVLA

> Modified from [UniAD](https://github.com/OpenDriveLab/UniAD) and [GPT-Driver](https://github.com/PointsCoder/GPT-Driver).

## nuScenes
Download nuScenes V1.0 full dataset data, CAN bus and map(v1.3) extensions [HERE](https://www.nuscenes.org/download), then follow the steps below to prepare the data.

### Download nuScenes, CAN_bus and Map extensions

```shell
cd DriveVLA
mkdir data
# Download nuScenes V1.0 full dataset data directly (or soft link) to data/
# Download CAN_bus and Map(v1.3) extensions directly (or soft link) to data/nuscenes/
```

### Download UniAD data info

```shell
cd DriveVLA/data
mkdir infos && cd infos
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_train.pkl  # train_infos
wget https://github.com/OpenDriveLab/UniAD/releases/download/v1.0/nuscenes_infos_temporal_val.pkl  # val_infos
```

### Download cached nuScenes information

We use the pre-cached nuScenes information `cached_nuscenes_info.pkl` following [GPT-Driver](https://github.com/PointsCoder/GPT-Driver). The cached data can be downloaded at [Google Drive](https://drive.google.com/drive/folders/1hUb1dsaDUABbUKnhj63vQBi0n4AZaZyM?usp=sharing) or using the following command.

```shell
pip install gdown
cd DriveVLA/data/nuscenes
gdown 16X0_-v-iXP9hVLNaDMmIiGhZKj24YOnb
```

### The Overall Structure

```shell
DriveVLA
├── data/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
│   │   ├── v1.0-trainval/
│   │   ├── cached_nuscenes_info.pkl
```

## Evaluation Dataset

We adopt the GT cache from [GPT-Driver](https://github.com/PointsCoder/GPT-Driver). Download gt for evaluation at [Google Drive](https://drive.google.com/drive/folders/1NCqPtdK8agPi1q3sr9-8-vPdYj08OCAE).

The structure is as follows:

```shell
eval_share
├── gt
│   ├── gt_traj_mask.pkl
│   ├── gt_traj.pkl
│   ├── planing_gt_segmentation_val
│   └── vad_gt_seg.pkl
├── __init__.py
├── metric.py
└── README.md
```
