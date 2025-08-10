"""
# Adapted from https://huggingface.co/MILVLG/imp-v1-3b/blob/main/vision_encoder.py
"""

from typing import Optional, Union
import torch
import torch.utils.checkpoint
from torch import nn
import os
import os.path as osp

from transformers.modeling_utils import PreTrainedModel
from transformers import PretrainedConfig
from llava.utils import rank0_print

from mmengine import Config
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger('shapely.geos').setLevel(logging.ERROR)

class UniadTrackMapConfig(PretrainedConfig):
    model_type = "uniad_track_map_model"

    def __init__(self, uniad_config_dict: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)

        self.uniad_config_dict = uniad_config_dict

class UniadTrackMapModel(PreTrainedModel):
    config_class = UniadTrackMapConfig
    base_model_prefix = "uniad_track_map"
    supports_gradient_checkpointing = True
    main_input_name = "pixel_values"
    _no_split_modules = ["UniAD"]

    def __init__(self, config: UniadTrackMapConfig, load_mmdet3d_weights=False, vision_tower_test_mode=False):
        super().__init__(config)

        self.config = config
        self.load_mmdet3d_weights = load_mmdet3d_weights
        self.vision_tower_test_mode = vision_tower_test_mode
        # build the UniAD model
        self.vision_model = self.build_uniad_track_map_model()

    def build_uniad_track_map_model(self):
        uniad_config_mmlab = Config()
        uniad_config_mmlab.merge_from_dict(self.config.uniad_config_dict)
        # import modules from plguin/xx, registry will be updated
        if hasattr(uniad_config_mmlab, 'plugin'):
            if uniad_config_mmlab.plugin:
                import importlib
                plugin_dir = uniad_config_mmlab.plugin_dir
                _module_dir = osp.dirname(plugin_dir)
                _module_dir = str(_module_dir).split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

        uniad_config_mmlab.model.pretrained = None
        uniad_config_mmlab.model.train_cfg = None
        model = build_model(uniad_config_mmlab.model, test_cfg=uniad_config_mmlab.get('test_cfg'))

        if self.load_mmdet3d_weights:
            checkpoint = load_checkpoint(model, 'checkpoints/uniad_base_track_map.pth', map_location='cpu')

            if 'CLASSES' in checkpoint.get('meta', {}):
                model.CLASSES = checkpoint['meta']['CLASSES']
            if 'PALETTE' in checkpoint.get('meta', {}):
                model.PALETTE = checkpoint['meta']['PALETTE']

        return model

    def _init_weights(self, module):
        """Initialize the weights"""
        pass

    def forward(self, data):
        if self.vision_tower_test_mode:
            _, results_for_vlm = self.vision_model(return_loss=False, rescale=True, **data)
        else:
            _, results_for_vlm = self.vision_model(return_loss=True, rescale=True, **data)
        return results_for_vlm

class UniadTrackMapVisionTower(nn.Module):
    def __init__(self, vision_tower, vision_tower_cfg, delay_load=False):
        super().__init__()

        uniad_config_dict = Config.fromfile('projects/configs/stage1_track_map/base_track_map.py').to_dict()
        self.config = UniadTrackMapConfig(uniad_config_dict=uniad_config_dict)

        self.vision_tower_name = vision_tower
        self.vision_tower: nn.Module = None
        self.is_loaded = False
        self.vision_tower_pretrained = vision_tower_cfg.vision_tower_pretrained
        if hasattr(vision_tower_cfg, "vision_tower_test_mode"):
            self.vision_tower_test_mode = vision_tower_cfg.vision_tower_test_mode
        else:  # set to False when in training mode
            self.vision_tower_test_mode = False

        self.image_processor = None

        if not delay_load:
            rank0_print(f"Loading vision tower: {vision_tower}")
            self.load_model()

        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()

        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(f"The checkpoint seems to contain `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`.")
            self.load_model()

        else:
            self.cfg_only = self.config

    def load_model(self, device_map="auto"):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.".format(self.vision_tower_name))
            return

        if self.vision_tower_pretrained:
            # Check if vision_tower_name points to a valid pretrained model path
            load_from_transformers_pretrained = (
                isinstance(self.vision_tower_pretrained, str) and 
                (os.path.exists(self.vision_tower_pretrained) or 
                self.vision_tower_pretrained.startswith('https://') or
                self.vision_tower_pretrained.startswith('http://'))
            )

            if load_from_transformers_pretrained:
                rank0_print(f"Loading UniAD transformers checkpoint from: {self.vision_tower_pretrained}")
                self.vision_tower = UniadTrackMapModel.from_pretrained(
                    self.vision_tower_pretrained, 
                    device_map=device_map
                )
                self.vision_tower.vision_tower_test_mode = self.vision_tower_test_mode
            else: # load from mmdet3d checkpoint
                rank0_print("Loading UniAD from mmdet3d checkpoint")
                self.vision_tower = UniadTrackMapModel(self.config, load_mmdet3d_weights=True, vision_tower_test_mode=self.vision_tower_test_mode)
        else:
            # only build the model, not load the weights
            rank0_print("Building UniAD from config. Weights will be loaded from the llava checkpoint.")
            self.vision_tower = UniadTrackMapModel(self.config, load_mmdet3d_weights=False, vision_tower_test_mode=self.vision_tower_test_mode)

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, data):
        return self.vision_tower(data)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device
