#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, SCENE_TOKEN_INDEX, TRACK_TOKEN_INDEX, MAP_TOKEN_INDEX, OBJECT_TOKEN_INDEX

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.vision_tower)
            if hasattr(self.vision_tower, "config"):
                # self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)
                self.mm_projector_track = build_vision_projector(config, vision_cfg=self.vision_tower.config)
                self.mm_projector_scene = build_vision_projector(config, vision_cfg=self.vision_tower.config)
                self.mm_projector_map = build_vision_projector(config, vision_cfg=self.vision_tower.config)
            else:
                self.mm_projector = build_vision_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # breakpoint()
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_mm_mlp_adapter_track = model_args.pretrain_mm_mlp_adapter_track
        pretrain_mm_mlp_adapter_scene = model_args.pretrain_mm_mlp_adapter_scene
        pretrain_mm_mlp_adapter_map = model_args.pretrain_mm_mlp_adapter_map
        mm_patch_merge_type = model_args.mm_patch_merge_type
        mm_hidden_size = model_args.mm_hidden_size

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = ""

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)
            vision_resampler = build_vision_resampler(model_args, vision_tower=vision_tower)
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.vision_tower = vision_tower
                self.vision_resampler = vision_resampler
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_resampler = self.vision_resampler[0]
                vision_tower = self.vision_tower[0]
            else:
                vision_resampler = self.vision_resampler
                vision_tower = self.vision_tower
            vision_tower.load_model()

            # In case it is frozen by LoRA
            for p in self.vision_resampler.parameters():
                p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", mm_hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.mm_hidden_size = mm_hidden_size
        if getattr(self, "mm_projector_track", None) is None:
            self.mm_projector_track = build_vision_projector(self.config)
        if getattr(self, "mm_projector_scene", None) is None:
            self.mm_projector_scene = build_vision_projector(self.config)
        if getattr(self, "mm_projector_map", None) is None:
            self.mm_projector_map = build_vision_projector(self.config)
        # if getattr(self, "mm_projector", None) is None:
        #     self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

        #     if "unpad" in mm_patch_merge_type:
        #         embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
        #         self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        # else:
        #     # In case it is frozen by LoRA
        #     for p in self.mm_projector.parameters():
        #         p.requires_grad = True

        def get_w(weights, keyword):
            return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}
        
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            try:
                incompatible_keys = self.mm_projector_track.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            except Exception as e:
                projector_weights = get_w(mm_projector_weights, "mm_projector")
                if "0.weight" in projector_weights:
                    pretrained_shape = projector_weights["0.weight"].shape
                    current_shape = self.mm_projector_track.state_dict()["0.weight"].shape
                    
                    if pretrained_shape != current_shape:
                        rank0_print(f"Warning: mm_projector_track weight shape mismatch. Pretrained: {pretrained_shape}, Current: {current_shape}")
                        projector_weights.pop("0.weight", None)
                        projector_weights.pop("0.bias", None)

                # Xavier initialization for first layer of projector
                if hasattr(self.mm_projector_track, "0"):
                    nn.init.xavier_uniform_(self.mm_projector_track.state_dict()["0.weight"])
                    if "0.bias" in self.mm_projector_track.state_dict():
                        nn.init.zeros_(self.mm_projector_track.state_dict()["0.bias"])
                
                incompatible_keys = self.mm_projector_track.load_state_dict(projector_weights, strict=False)

            rank0_print(f"Loaded mm projector track weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

            try:
                incompatible_keys = self.mm_projector_scene.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            except Exception as e:
                projector_weights = get_w(mm_projector_weights, "mm_projector")
                if "0.weight" in projector_weights:
                    pretrained_shape = projector_weights["0.weight"].shape
                    current_shape = self.mm_projector_scene.state_dict()["0.weight"].shape
                    
                    if pretrained_shape != current_shape:
                        rank0_print(f"Warning: mm_projector_scene weight shape mismatch. Pretrained: {pretrained_shape}, Current: {current_shape}")
                        projector_weights.pop("0.weight", None)
                        projector_weights.pop("0.bias", None)

                # Xavier initialization for first layer of projector
                if hasattr(self.mm_projector_scene, "0"):
                    nn.init.xavier_uniform_(self.mm_projector_scene.state_dict()["0.weight"])
                    if "0.bias" in self.mm_projector_scene.state_dict():
                        nn.init.zeros_(self.mm_projector_scene.state_dict()["0.bias"])
                
                incompatible_keys = self.mm_projector_scene.load_state_dict(projector_weights, strict=False)

            rank0_print(f"Loaded mm projector scene weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

            try:
                incompatible_keys = self.mm_projector_map.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            except Exception as e:
                projector_weights = get_w(mm_projector_weights, "mm_projector")
                if "0.weight" in projector_weights:
                    pretrained_shape = projector_weights["0.weight"].shape
                    current_shape = self.mm_projector_map.state_dict()["0.weight"].shape
                    
                    if pretrained_shape != current_shape:
                        rank0_print(f"Warning: mm_projector_map weight shape mismatch. Pretrained: {pretrained_shape}, Current: {current_shape}")
                        projector_weights.pop("0.weight", None)
                        projector_weights.pop("0.bias", None)

                # Xavier initialization for first layer of projector
                if hasattr(self.mm_projector_map, "0"):
                    nn.init.xavier_uniform_(self.mm_projector_map.state_dict()["0.weight"])
                    if "0.bias" in self.mm_projector_map.state_dict():
                        nn.init.zeros_(self.mm_projector_map.state_dict()["0.bias"])
                
                incompatible_keys = self.mm_projector_map.load_state_dict(projector_weights, strict=False)

            rank0_print(f"Loaded mm projector map weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")

        if pretrain_mm_mlp_adapter_track is not None:
            mm_projector_track_weights = torch.load(pretrain_mm_mlp_adapter_track, map_location="cpu")
            incompatible_keys = self.mm_projector_track.load_state_dict(get_w(mm_projector_track_weights, "mm_projector_track"), strict=True)
            rank0_print(f"Loaded mm projector track weights from {pretrain_mm_mlp_adapter_track}. Incompatible keys: {incompatible_keys}")

        if pretrain_mm_mlp_adapter_scene is not None:
            mm_projector_scene_weights = torch.load(pretrain_mm_mlp_adapter_scene, map_location="cpu")
            incompatible_keys = self.mm_projector_scene.load_state_dict(get_w(mm_projector_scene_weights, "mm_projector_scene"), strict=True)
            rank0_print(f"Loaded mm projector scene weights from {pretrain_mm_mlp_adapter_scene}. Incompatible keys: {incompatible_keys}")

        if pretrain_mm_mlp_adapter_map is not None:
            mm_projector_map_weights = torch.load(pretrain_mm_mlp_adapter_map, map_location="cpu")
            incompatible_keys = self.mm_projector_map.load_state_dict(get_w(mm_projector_map_weights, "mm_projector_map"), strict=True)
            rank0_print(f"Loaded mm projector map weights from {pretrain_mm_mlp_adapter_map}. Incompatible keys: {incompatible_keys}")

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_2dPool(self, image_feature):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / 2), math.ceil(weight / 2)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []

        for idx, feat in enumerate(per_videos_or_images_features):
            feat = self.get_model().mm_projector(feat)
            if idx in video_idx_in_batch:
                feat = self.get_2dPool(feat)
            all_videos_or_images_features.append(feat)
        return all_videos_or_images_features
    
    def encode_vision_tower_result(self, vision_tower_result):

        result_track = vision_tower_result["result_track"]
        result_seg = vision_tower_result["result_seg"]

        track_query_embeddings = result_track["track_query_embeddings"]  # [20+, 256]
        chosen_output_query_things = result_seg["chosen_output_query_things"]  # [10+, 256]
        output_query_stuff = result_seg['output_query_stuff']  # [1, 256]
        map_seg_query_embeddings = torch.cat([chosen_output_query_things, output_query_stuff], dim=0)  # [10+, 256]

        # print(f"\n>>> track_query_embeddings: {track_query_embeddings.shape}")
        # print(f">>> map_seg_query_embeddings: {map_seg_query_embeddings.shape}")

        # # use the bev feature to encode scene
        # bev_embed = result_track["bev_embed"].squeeze()  # [40000, 256]
        # # Apply pooling to bev_embed [40000, 256] -> [625, 256]
        # # Reshape to [1, 256, 200, 200]
        # bev_embed = bev_embed.transpose(0, 1)  # [256, 40000]
        # bev_embed = bev_embed.view(1, 256, 200, 200)  # [1, 256, 200, 200]
        
        # # Apply pooling 3 times to reduce spatial dimensions by 8x
        # bev_embed = F.avg_pool2d(bev_embed, kernel_size=2, stride=2)  # [1, 256, 100, 100]
        # bev_embed = F.avg_pool2d(bev_embed, kernel_size=2, stride=2)  # [1, 256, 50, 50]
        # bev_embed = F.avg_pool2d(bev_embed, kernel_size=2, stride=2)  # [1, 256, 25, 25]
        
        # # Reshape back to [625, 256]
        # bev_embed = bev_embed.squeeze().permute(1, 2, 0).reshape(-1, 256)
        # scene_feature = self.get_model().mm_projector_scene(bev_embed)

        # use the image feature to encode scene
        img_feat_2D = result_track["img_feat_2D"]  # [1, 6, 256, 15, 25]
        img_feat_2D = img_feat_2D.squeeze(0)  # [6, 256, 15, 25]
        img_feat_2D = F.adaptive_max_pool2d(img_feat_2D, (3, 5))  # [6, 256, 3, 5]
        img_feat_2D = img_feat_2D.flatten(2) # [6, 256, 15]
        img_feat_2D = img_feat_2D.transpose(1, 2)  # [6, 15, 256]
        img_feat_2D = img_feat_2D.reshape(-1, 256)  # [90, 256]
        scene_feature = self.get_model().mm_projector_scene(img_feat_2D)

        if track_query_embeddings is not None:
            track_feature = self.get_model().mm_projector_track(track_query_embeddings.to(dtype=img_feat_2D.dtype))
        else:
            track_feature = None

        map_feature = self.get_model().mm_projector_map(map_seg_query_embeddings.to(dtype=img_feat_2D.dtype))

        return scene_feature, track_feature, map_feature
    
    def prepare_inputs_labels_for_multimodal_uniad_vlm(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None, uniad_data=None, uniad_pth=None, qa_instance_ind=None):
        vision_tower = self.get_vision_tower()

        if (images is None and uniad_pth is None and uniad_data is None) or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if uniad_pth is not None:
            vision_tower_result = uniad_pth
        elif uniad_data is not None:
            vision_tower_result = vision_tower(uniad_data)
            vision_tower_test_mode = vision_tower.vision_tower.vision_tower_test_mode
        else:
            vision_tower_result = None

        if vision_tower_result is not None:
            scene_feature, track_feature, map_feature = self.encode_vision_tower_result(vision_tower_result)
            if qa_instance_ind is not None:
                track_gt_inds_to_embed_idx: Dict[int, int] = vision_tower_result["result_track"]["track_gt_inds_to_embed_idx"]
                qa_instance_track_embed_idx = track_gt_inds_to_embed_idx.get(qa_instance_ind, None)
                if qa_instance_track_embed_idx is not None:
                    qa_instance_track_feature = track_feature[qa_instance_track_embed_idx].unsqueeze(0)
                else:
                    print(f"Warning: qa_instance_ind {qa_instance_ind} not found in track_gt_inds_to_embed_idx")
        else:
            scene_feature = track_feature = map_feature = None

        if images is not None:
            if type(images) is list or images.ndim == 5:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

                video_idx_in_batch = []
                for _ in range(len(modalities)):
                    if modalities[_] == "video":
                        video_idx_in_batch.append(_)

                images_list = []
                for image in images:
                    if image.ndim == 4:
                        images_list.append(image)
                    else:
                        images_list.append(image.unsqueeze(0))
                
                concat_images = torch.cat([image for image in images_list], dim=0)
                split_sizes = [image.shape[0] for image in images_list]
                encoded_image_features = self.encode_images(concat_images)

                # This is a list, each element is [num_images, patch * patch, dim]
                # rank_print(f"Concat images : {concat_images.shape}")
                encoded_image_features = torch.split(encoded_image_features, split_sizes)
                image_features = []
                for idx, image_feat in enumerate(encoded_image_features):
                    if idx in video_idx_in_batch:
                        image_features.append(self.get_2dPool(image_feat))     # [16,729, 3584] --> [16, 196, 3584]
                    else:
                        image_features.append(image_feat)
                # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
                # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
                # image_features = torch.split(image_features, split_sizes, dim=0)
                mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
                image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
                # breakpoint()

                if mm_patch_merge_type == "flat":
                    image_features = [x.flatten(0, 1) for x in image_features]
                elif mm_patch_merge_type.startswith("spatial"):
                    new_image_features = []
                    for image_idx, image_feature in enumerate(image_features):
                        # FIXME: now assume the image is square, and split to 2x2 patches
                        # num_patches = h * w, where h = w = sqrt(num_patches)
                        # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                        # we want to first unflatten it to (2, 2, h, w, hidden_size)
                        # rank0_print("At least we are reaching here")
                        if image_idx in video_idx_in_batch:  # video operations
                            # rank0_print("Video")
                            if "unpad" in mm_patch_merge_type:
                                # image_feature = image_feature.permute(2, 0, 1).contiguous()
                                # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                # image_feature = image_feature.permute(1, 2, 0).contiguous()
                                image_feature = image_feature.flatten(0, 1)
                                image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0)

                        elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                            # rank0_print("Single-images")
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]

                            if "anyres_max" in image_aspect_ratio:
                                matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                                if matched_anyres_max_num_patches:
                                    max_num_patches = int(matched_anyres_max_num_patches.group(1))

                            if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                                if hasattr(self.get_vision_tower(), "image_size"):
                                    vision_tower_image_size = self.get_vision_tower().image_size
                                else:
                                    raise ValueError("vision_tower_image_size is not found in the vision tower.")
                                try:
                                    num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                                except Exception as e:
                                    rank0_print(f"Error: {e}")
                                    num_patch_width, num_patch_height = 2, 2
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                # image_feature = image_feature.view(2, 2, height, width, -1)
                                # change it to [1,5] for DriveGPT inference
                                image_feature = image_feature.view(1, 5, height, width, -1)


                            if "maxpool2x2" in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = nn.functional.max_pool2d(image_feature, 2)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                                unit = image_feature.shape[2] # image_feature torch.Size([2, 4, 27, 27, 896]), [2,4] are path_nums, 27,27 are patch size, 896 is hidden size
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # torch.Size([896, 2, 27, 4, 27])
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3) # torch.Size([896, 54, 108])
                                image_feature = unpad_image(image_feature, image_sizes[image_idx]) # torch.Size([896, 54, 96])
                                c, h, w = image_feature.shape
                                times = math.sqrt(h * w / (max_num_patches * unit**2))
                                if times > 1.1:
                                    image_feature = image_feature[None]
                                    image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                                image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1) # --> torch.Size([896, 54, 97])
                                #  self.model.image_newline.shape torch.Size([896])
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1) # torch.Size([5238, 896])
                            elif "unpad" in mm_patch_merge_type:
                                # breakpoint()

                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                                image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            if "nobase" in mm_patch_merge_type:
                                pass
                            else:
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            new_image_features.append(image_feature)
                        else:  # single image operations
                            image_feature = image_feature[0]
                            if "unpad" in mm_patch_merge_type:
                                image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                        new_image_features.append(image_feature)
                    image_features = new_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            else:
                image_features = self.encode_images(images)
        else:
            image_features = []

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # breakpoint()
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # Count all special tokens
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            num_scenes = (cur_input_ids == SCENE_TOKEN_INDEX).sum()
            num_tracks = (cur_input_ids == TRACK_TOKEN_INDEX).sum() 
            num_maps = (cur_input_ids == MAP_TOKEN_INDEX).sum()
            num_objects = (cur_input_ids == OBJECT_TOKEN_INDEX).sum()

            if num_images == 0 and num_scenes == 0 and num_tracks == 0 and num_maps == 0 and num_objects == 0:
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                if images is not None:
                    cur_image_features = image_features[cur_image_idx]
                    cur_input_embeds = torch.cat([cur_input_embeds, cur_image_features[0:0]], dim=0)
                    cur_image_idx += 1
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                continue

            # Get all special token positions
            special_token_indices = [-1]
            special_token_indices.extend(torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist())
            special_token_indices.extend(torch.where(cur_input_ids == SCENE_TOKEN_INDEX)[0].tolist())
            special_token_indices.extend(torch.where(cur_input_ids == TRACK_TOKEN_INDEX)[0].tolist())
            special_token_indices.extend(torch.where(cur_input_ids == MAP_TOKEN_INDEX)[0].tolist())
            special_token_indices.extend(torch.where(cur_input_ids == OBJECT_TOKEN_INDEX)[0].tolist())
            special_token_indices.sort()
            special_token_indices.append(cur_input_ids.shape[0])

            # Split input ids and labels
            cur_input_ids_nospecial = []
            cur_labels = labels[batch_idx]
            cur_labels_nospecial = []
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_nospecial.append(cur_input_ids[special_token_indices[i] + 1 : special_token_indices[i + 1]])
                cur_labels_nospecial.append(cur_labels[special_token_indices[i] + 1 : special_token_indices[i + 1]])

            # Get embeddings for text tokens
            split_sizes = [x.shape[0] for x in cur_labels_nospecial]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospecial))
            cur_input_embeds_no_special = torch.split(cur_input_embeds, split_sizes, dim=0)

            # Insert features at corresponding positions
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_image_idx = 0

            for i in range(len(special_token_indices) - 1):
                # Add text embeddings
                cur_new_input_embeds.append(cur_input_embeds_no_special[i])
                cur_new_labels.append(cur_labels_nospecial[i])
                
                if special_token_indices[i + 1] < cur_input_ids.shape[0]:
                    token_type = cur_input_ids[special_token_indices[i + 1]]
                    
                    # Add corresponding feature based on token type
                    if token_type == IMAGE_TOKEN_INDEX:
                        if images is not None:
                            cur_image_features = image_features[cur_image_idx]
                            cur_image_idx += 1
                            cur_new_input_embeds.append(cur_image_features)
                            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif token_type == SCENE_TOKEN_INDEX:
                        cur_new_input_embeds.append(scene_feature)
                        cur_new_labels.append(torch.full((scene_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif token_type == TRACK_TOKEN_INDEX:
                        if track_feature is not None:
                            cur_new_input_embeds.append(track_feature)
                            cur_new_labels.append(torch.full((track_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif token_type == MAP_TOKEN_INDEX:
                        cur_new_input_embeds.append(map_feature)
                        cur_new_labels.append(torch.full((map_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif token_type == OBJECT_TOKEN_INDEX:
                        if qa_instance_track_embed_idx is not None:
                            cur_new_input_embeds.append(qa_instance_track_feature)
                            cur_new_labels.append(torch.full((qa_instance_track_feature.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        # breakpoint()
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]
            encoded_image_features = self.encode_images(concat_images)

            # This is a list, each element is [num_images, patch * patch, dim]
            # rank_print(f"Concat images : {concat_images.shape}")
            encoded_image_features = torch.split(encoded_image_features, split_sizes)
            image_features = []
            for idx, image_feat in enumerate(encoded_image_features):
                if idx in video_idx_in_batch:
                    image_features.append(self.get_2dPool(image_feat))     # [16,729, 3584] --> [16, 196, 3584]
                else:
                    image_features.append(image_feat)
            # image_features = self.encode_multimodals(concat_images, video_idx_in_batch, split_sizes)
            # rank_print(f"Encoded image feats : {[x.shape for x in image_features]}")
            # image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            # breakpoint()

            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        if "unpad" in mm_patch_merge_type:
                            # image_feature = image_feature.permute(2, 0, 1).contiguous()
                            # image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            # image_feature = image_feature.permute(1, 2, 0).contiguous()
                            image_feature = image_feature.flatten(0, 1)
                            image_feature = torch.cat((image_feature, self.model.image_newline[None].to(image_feature.device)), dim=0)

                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images")
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size)
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                num_patch_width, num_patch_height = 2, 2
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            # image_feature = image_feature.view(2, 2, height, width, -1)
                            # change it to [1,5] for DriveGPT inference
                            image_feature = image_feature.view(1, 5, height, width, -1)


                        if "maxpool2x2" in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            unit = image_feature.shape[2] # image_feature torch.Size([2, 4, 27, 27, 896]), [2,4] are path_nums, 27,27 are patch size, 896 is hidden size
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous() # torch.Size([896, 2, 27, 4, 27])
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3) # torch.Size([896, 54, 108])
                            image_feature = unpad_image(image_feature, image_sizes[image_idx]) # torch.Size([896, 54, 96])
                            c, h, w = image_feature.shape
                            times = math.sqrt(h * w / (max_num_patches * unit**2))
                            if times > 1.1:
                                image_feature = image_feature[None]
                                image_feature = nn.functional.interpolate(image_feature, [int(h // times), int(w // times)], mode="bilinear")[0]
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1) # --> torch.Size([896, 54, 97])
                            #  self.model.image_newline.shape torch.Size([896])
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1) # torch.Size([5238, 896])
                        elif "unpad" in mm_patch_merge_type:
                            # breakpoint()

                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        # breakpoint()
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            # breakpoint()
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))  # TODO: check if multiple images are used in training
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        # add [batch_idx] here for drivePGT
                        cur_image_features = image_features[cur_image_idx]  # TODO: check if multiple images are used in training
                    except IndexError:
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False