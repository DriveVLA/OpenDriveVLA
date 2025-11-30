# OpenDriveVLA: Towards End-to-end Autonomous Driving with  Large Vision Language Action Model

<h3 align="center">
  <a href="https://drivevla.github.io/">Project Page</a> |
  <a href="https://arxiv.org/abs/2503.23463">arXiv</a>
</h3>

![](assets/drivevla-ModelArc.jpg)

## Overview ✨

- [Todo List](#todo-list-)
- [News](#news-)
- [Getting Started](#getting-started-)
- [Citation](#citation-)

## TODO List 📅

We will release the model code and checkpoints soon. Stay tuned! 🔥

- [x] Release environment setup
- [x] Release inference code
- [x] Release checkpoints
- [ ] Release training scripts


## News 📢

- **`2025/11/14`** Released the OpenDriveVLA 0.5B checkpoint on [Hugging Face](https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B). 🌟
- **`2025/11/08`** OpenDriveVLA paper accepted by AAAI 2026. 🎉
- **`2025/08/10`** OpenDriveVLA model & inference code released. 🔥
- **`2025/04/01`** OpenDriveVLA [paper](https://arxiv.org/abs/2503.23463) is available on arXiv.
- **`2025/03/28`** We release the environment setup of OpenDriveVLA.
  - To make the dependencies of our OpenDriveVLA model [[mmcv](https://github.com/open-mmlab/mmcv) & [mmdet3d](https://github.com/open-mmlab/mmdetection3d)] compatible with [PyTorch 2.1.2](https://pytorch.org/) and support [Transformers](https://github.com/huggingface/transformers) and [Deepspeed](https://github.com/deepspeedai/DeepSpeed), we selected specific versions and enhanced the source code accordingly. The resulting customized libraries are available in the `third_party` folder.

## Getting Started 🌟

1. [Environment Installation](docs/1_INSTALL.md)
2. [Data Preparation](docs/2_DATA_PREP.md)
3. [Inference & Evaluatation](docs/3_EVAL.md)

## Citation 📝

If you find our project useful for your research, please consider citing our paper and codebase with the following BibTeX:

```bibtex
@misc{zhou2025opendrivevlaendtoendautonomousdriving,
      title={OpenDriveVLA: Towards End-to-end Autonomous Driving with Large Vision Language Action Model}, 
      author={Xingcheng Zhou and Xuyuan Han and Feng Yang and Yunpu Ma and Volker Tresp and Alois Knoll},
      year={2025},
      eprint={2503.23463},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.23463}, 
}
```

## Acknowledgement 🤝

- [Transformers](https://github.com/huggingface/transformers)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [mmcv](https://github.com/open-mmlab/mmcv)
- [mmdet3d](https://github.com/open-mmlab/mmdetection3d)
- [GPT-Driver](https://github.com/PointsCoder/GPT-Driver)
- [Hint-AD](https://github.com/Robot-K/Hint-AD)
- [TOD3Cap](https://github.com/jxbbb/TOD3Cap)
