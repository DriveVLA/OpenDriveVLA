# Install DriveVLA

## 1. Install LLaVA

```shell
conda create -n drivevla python=3.10 -y
conda activate drivevla
pip install --upgrade pip  # Enable PEP 660 support.
pip install torch==2.1.2
# pip install -e ".[train]"
```

## 2. Install mmcv-full from source code

### (1) GCC: Make sure gcc>=5 in conda env

```shell
# If gcc is not installed:
# conda install -c omgarcia gcc-6 # gcc-6.2

export PATH=YOUR_GCC_PATH/bin:$PATH
# Eg: export PATH=/mnt/gcc-5.4/bin:$PATH
```

### (2) CUDA: Before installing MMCV family, you need to set up the CUDA_HOME (for compiling some operators on the gpu)

```shell
export CUDA_HOME=YOUR_CUDA_PATH/
# Eg: export CUDA_HOME=/mnt/cuda-11.1/
```

### (3) Install mmcv-full from source code

```shell
cd DriveVLA # cd to the root of this repo
cd third_party/mmcv_1_7_2
pip install -r requirements/optional.txt
MMCV_WITH_OPS=1 pip install .
```

## 3. Install mmdet and mmseg

```shell
pip install mmdet==2.26.0 mmsegmentation==0.29.1 mmengine==0.9.0 motmetrics==1.4.0 casadi==3.6.0
```

## 4. Install mmdet3d from source code

```shell
cd DriveVLA # cd to the root of this repo
cd third_party/mmdetection3d_1_0_0rc6
pip install scipy==1.10.1 scikit-image==0.19.3 fsspec
pip install .
```

## 5. Troubleshoot

### If you encounter the error: `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`

```shell
apt-get update && apt-get install libgl1
```

### If you encounter the error: `Error code: libgfortran.so.5: cannot open shared object file: No such file or directory`

```shell
sudo apt-get install libgfortran5
# or
conda install libgfortran
```
