#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
import copy
import os
from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from mmdet.models.builder import build_head
# from drivevla.utils.iou import get_3d_iou
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D
import scipy.optimize as opt

@DETECTORS.register_module()
class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """
    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(
            track=1.0,
            map=1.0,
            motion=1.0,
            occ=1.0,
            planning=1.0
        ),
        **kwargs,
    ):
        super(UniAD, self).__init__(**kwargs)
        if seg_head:
            self.seg_head = build_head(seg_head)
        if occ_head:
            self.occ_head = build_head(occ_head)
        if motion_head:
            self.motion_head = build_head(motion_head)
        if planning_head:
            self.planning_head = build_head(planning_head)
        
        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == \
               {'track', 'occ', 'motion', 'map', 'planning'}

    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None
    
    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
        

    # Add the subtask loss to the whole model loss
    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      
                      # Occ_gt
                      gt_segmentation=None,
                      gt_instance=None, 
                      gt_occ_img_is_valid=None,
                      
                      #planning
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      
                      # fut gt for planning
                      gt_future_boxes=None,
                      **kwargs,  # [1, 9]
                      ):
        """Forward training function for the model that includes multiple tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning.

            Args:
            img (torch.Tensor, optional): Tensor containing images of each sample with shape (N, C, H, W). Defaults to None.
            img_metas (list[dict], optional): List of dictionaries containing meta information for each sample. Defaults to None.
            gt_bboxes_3d (list[:obj:BaseInstance3DBoxes], optional): List of ground truth 3D bounding boxes for each sample. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): List of tensors containing ground truth labels for 3D bounding boxes. Defaults to None.
            gt_inds (list[torch.Tensor], optional): List of tensors containing indices of ground truth objects. Defaults to None.
            l2g_t (list[torch.Tensor], optional): List of tensors containing translation vectors from local to global coordinates. Defaults to None.
            l2g_r_mat (list[torch.Tensor], optional): List of tensors containing rotation matrices from local to global coordinates. Defaults to None.
            timestamp (list[float], optional): List of timestamps for each sample. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): List of tensors containing ground truth 2D bounding boxes in images to be ignored. Defaults to None.
            gt_lane_labels (list[torch.Tensor], optional): List of tensors containing ground truth lane labels. Defaults to None.
            gt_lane_bboxes (list[torch.Tensor], optional): List of tensors containing ground truth lane bounding boxes. Defaults to None.
            gt_lane_masks (list[torch.Tensor], optional): List of tensors containing ground truth lane masks. Defaults to None.
            gt_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth future trajectories. Defaults to None.
            gt_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth future trajectory masks. Defaults to None.
            gt_past_traj (list[torch.Tensor], optional): List of tensors containing ground truth past trajectories. Defaults to None.
            gt_past_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth past trajectory masks. Defaults to None.
            gt_sdc_bbox (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car bounding boxes. Defaults to None.
            gt_sdc_label (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car labels. Defaults to None.
            gt_sdc_fut_traj (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectories. Defaults to None.
            gt_sdc_fut_traj_mask (list[torch.Tensor], optional): List of tensors containing ground truth self-driving car future trajectory masks. Defaults to None.
            gt_segmentation (list[torch.Tensor], optional): List of tensors containing ground truth segmentation masks. Defaults to
            gt_instance (list[torch.Tensor], optional): List of tensors containing ground truth instance segmentation masks. Defaults to None.
            gt_occ_img_is_valid (list[torch.Tensor], optional): List of tensors containing binary flags indicating whether an image is valid for occupancy prediction. Defaults to None.
            sdc_planning (list[torch.Tensor], optional): List of tensors containing self-driving car planning information. Defaults to None.
            sdc_planning_mask (list[torch.Tensor], optional): List of tensors containing self-driving car planning masks. Defaults to None.
            command (list[torch.Tensor], optional): List of tensors containing high-level command information for planning. Defaults to None.
            gt_future_boxes (list[torch.Tensor], optional): List of tensors containing ground truth future bounding boxes for planning. Defaults to None.
            gt_future_labels (list[torch.Tensor], optional): List of tensors containing ground truth future labels for planning. Defaults to None.
            
            Returns:
                dict: Dictionary containing losses of different tasks, such as tracking, segmentation, motion prediction, occupancy prediction, and planning. Each key in the dictionary 
                    is prefixed with the corresponding task name, e.g., 'track', 'map', 'motion', 'occ', and 'planning'. The values are the calculated losses for each task.
        """
        losses = dict()
        len_queue = img.size(1)
        

        losses_track, outs_track = self.forward_track_train(img, gt_bboxes_3d, gt_labels_3d, gt_past_traj, gt_past_traj_mask, gt_inds, gt_sdc_bbox, gt_sdc_label,
                                                        l2g_t, l2g_r_mat, img_metas, timestamp)
        losses_track = self.loss_weighted_and_prefixed(losses_track, prefix='track')
        losses.update(losses_track)
        
        # Upsample bev for tiny version
        outs_track = self.upsample_bev_if_tiny(outs_track)

        bev_embed = outs_track["bev_embed"]
        bev_pos  = outs_track["bev_pos"]

        img_metas = [each[len_queue-1] for each in img_metas]

        outs_seg = dict()
        if self.with_seg_head:          
            losses_seg, outs_seg = self.seg_head.forward_train(bev_embed, img_metas,
                                                          gt_lane_labels, gt_lane_bboxes, gt_lane_masks)
            
            losses_seg = self.loss_weighted_and_prefixed(losses_seg, prefix='map')
            losses.update(losses_seg)

        outs_motion = dict()
        # Forward Motion Head
        if self.with_motion_head:
            ret_dict_motion = self.motion_head.forward_train(bev_embed,
                                                        gt_bboxes_3d, gt_labels_3d, 
                                                        gt_fut_traj, gt_fut_traj_mask, 
                                                        gt_sdc_fut_traj, gt_sdc_fut_traj_mask, 
                                                        outs_track=outs_track, outs_seg=outs_seg
                                                    )
            losses_motion = ret_dict_motion["losses"]
            outs_motion = ret_dict_motion["outs_motion"]
            outs_motion['bev_pos'] = bev_pos
            losses_motion = self.loss_weighted_and_prefixed(losses_motion, prefix='motion')
            losses.update(losses_motion)

        # Forward Occ Head
        if self.with_occ_head:
            if outs_motion['track_query'].shape[1] == 0:
                # TODO: rm hard code
                outs_motion['track_query'] = torch.zeros((1, 1, 256)).to(bev_embed)
                outs_motion['track_query_pos'] = torch.zeros((1,1, 256)).to(bev_embed)
                outs_motion['traj_query'] = torch.zeros((3, 1, 1, 6, 256)).to(bev_embed)
                outs_motion['all_matched_idxes'] = [[-1]]
            losses_occ = self.occ_head.forward_train(
                            bev_embed, 
                            outs_motion, 
                            gt_inds_list=gt_inds,
                            gt_segmentation=gt_segmentation,
                            gt_instance=gt_instance,
                            gt_img_is_valid=gt_occ_img_is_valid,
                        )
            losses_occ = self.loss_weighted_and_prefixed(losses_occ, prefix='occ')
            losses.update(losses_occ)
        

        # Forward Plan Head
        if self.with_planning_head:
            outs_planning = self.planning_head.forward_train(bev_embed, outs_motion, sdc_planning, sdc_planning_mask, command, gt_future_boxes)
            losses_planning = outs_planning['losses']
            losses_planning = self.loss_weighted_and_prefixed(losses_planning, prefix='planning')
            losses.update(losses_planning)
        
        for k,v in losses.items():
            losses[k] = torch.nan_to_num(v)

        results_for_vlm = self.get_results_for_vlm(img_metas[0], outs_track, outs_seg[0], sdc_planning[0], sdc_planning_mask[0], command[0], in_uniad_train=True, **kwargs)

        return [], results_for_vlm
    
    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight[prefix]
        loss_dict = {f"{prefix}.{k}" : v*loss_factor for k, v in loss_dict.items()}
        return loss_dict

    def forward_test(self,
                     img=None,
                     img_metas=None,
                     l2g_t=None,
                     l2g_r_mat=None,
                     timestamp=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     rescale=False,
                     # planning gt(for evaluation only)
                     sdc_planning=None,
                     sdc_planning_mask=None,
                     command=None,
 
                     # Occ_gt (for evaluation only)
                     gt_segmentation=None,
                     gt_instance=None, 
                     gt_occ_img_is_valid=None,
                     **kwargs
                    ):
        """Test function
        """
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        # first frame
        if self.prev_frame_info['scene_token'] is None:
            img_metas[0][0]['can_bus'][:3] = 0
            img_metas[0][0]['can_bus'][-1] = 0
        # following frames
        else:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # Upsample bev for tiny model        
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])
        
        bev_embed = result_track[0]["bev_embed"]

        if self.with_seg_head:
            result_seg =  self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(bev_embed, outs_track=result_track[0], outs_seg=result_seg[0])
            outs_motion['bev_pos'] = result_track[0]['bev_pos']

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            outs_occ = self.occ_head.forward_test(
                bev_embed, 
                outs_motion,
                no_query = occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]['occ'] = outs_occ
        
        if self.with_planning_head:
            planning_gt=dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)
            result[0]['planning'] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        results_for_vlm = self.get_results_for_vlm(img_metas[0], result_track[0], result_seg[0], sdc_planning[0], sdc_planning_mask[0], command[0], in_uniad_train=False, **kwargs)

        pop_track_list = ['prev_bev', 'bev_pos', 'bev_embed', 'track_query_embeddings', 'sdc_embedding', 'track_instances_fordet', 'img_feat_2D']
        seg_pop_list = ['args_tuple', 'ret_iou', 'output_query_things', 'output_query_stuff', 'chosen_output_query_things']
        occ_pop_list = ['seg_out_mask', 'flow_out', 'future_states_occ', 'pred_ins_masks', 'pred_raw_occ', 'pred_ins_logits', 'pred_ins_sigmoid']
        
        for i, res in enumerate(result):
            res['token'] = img_metas[i]['sample_idx']
            res.update({k: v for k, v in result_track[i].items() if k not in pop_track_list})
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update({k: v for k, v in result_seg[i].items() if k not in seg_pop_list})
            if self.with_occ_head:
                res['occ'] = {k: v for k, v in result[i]['occ'].items() if k not in occ_pop_list}

        # for key in ['segm', 'output_query_things', 'output_query_stuff', 'chosen_output_query_things']:
        #     result[0]['pts_bbox'].pop(key)

        # # save the result for visualization
        # output_dir = 'output/result_pth_for_vis/result'
        # os.makedirs(output_dir, exist_ok=True)
        # torch.save(result, os.path.join(output_dir, f"{result[0]['token']}.pth"))

        return result, results_for_vlm
    
    def get_results_for_vlm(self, img_metas, result_track, result_seg, sdc_planning, sdc_planning_mask, command, in_uniad_train=False, **kwargs):
        if not in_uniad_train:
            if 'gt_bboxes_3d' in kwargs:
                gt_bboxes_3d = kwargs['gt_bboxes_3d'][0][0][0].tensor
                gt_labels_3d = kwargs['gt_labels_3d'][0][0][0]
                gt_inds = kwargs['gt_inds'][0].squeeze(0)
                detected_bboxes = (result_track['track_bbox_results'][0][0].tensor).to(gt_bboxes_3d)[:-1, :7] # remove the last one ego track
                matched_idx, iou_matrix = match_bbox(detected_bboxes, gt_bboxes_3d[:,:7])
                track_gt_inds_to_embed_idx = {}
                for embed_idx, gt_inds_idx in enumerate(matched_idx):
                    if len(gt_inds) == 0:
                        print("gt_inds is empty")
                        continue
                    if gt_inds_idx != -1:
                        track_gt_inds_to_embed_idx[int(gt_inds[gt_inds_idx])] = embed_idx

                result_track['gt_bboxes_3d'] = gt_bboxes_3d
                result_track['gt_labels_3d'] = gt_labels_3d
                result_track['gt_inds'] = gt_inds
                result_track['iou_matrix'] = iou_matrix
                result_track['matched_idx'] = matched_idx
                result_track['track_gt_inds_to_embed_idx'] = track_gt_inds_to_embed_idx

        _img_metas = {}
        _result_track = {}
        _result_seg = {}

        save_keys_img_metas = ['filename', 'ori_shape', 'img_norm_cfg', 'sample_idx', 'prev_idx', 'next_idx', 'scene_token', 'can_bus']
        save_keys_result_seg = ['output_query_things', 'output_query_stuff', 'chosen_output_query_things']

        pop_keys_result_track = ['bev_pos']
        
        _img_metas.update({key: img_metas[key] for key in img_metas.keys() if key in save_keys_img_metas})

        _result_track.update({key: result_track[key] for key in result_track.keys() if key not in pop_keys_result_track})
        
        if in_uniad_train:
            _result_track.update({"track_query_embeddings_all": result_track["track_instances"].output_embedding})  
        else:
            _result_track.update({"track_query_embeddings_all": result_track["track_instances_fordet"].output_embedding})  
        
        # track_query_embeddings_all[900] -> hard_code: assume the 901 query is sdc query
        # remove the sdc query
        if "track_query_embeddings_all" in _result_track and _result_track["track_query_embeddings_all"] is not None:
            mask = torch.ones(_result_track["track_query_embeddings_all"].shape[0], dtype=torch.bool, 
                             device=_result_track["track_query_embeddings_all"].device)
            if mask.shape[0] > 900:
                mask[900] = False
                _result_track["track_query_embeddings_all"] = _result_track["track_query_embeddings_all"][mask]

        _result_seg.update({key: result_seg[key] for key in result_seg.keys() if key in save_keys_result_seg})

        if not in_uniad_train:
            if _result_track['track_query_embeddings'].shape[0] > 1:
                _result_track['track_query_embeddings'] = _result_track['track_query_embeddings'][:-1]
            else:
                _result_track['track_query_embeddings'] = None

        results_for_vlm = dict(
            scene_token=_img_metas['scene_token'],
            sample_token=_img_metas['sample_idx'],
            img_metas=_img_metas,
            result_track=_result_track,
            result_seg=_result_seg,
            planning_gt=dict(
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command
            )
        )
        return results_for_vlm

# assign each detected bbox with a ground truth bbox
def match_bbox(detected_bboxes, gt_bboxes):
    if len(gt_bboxes) == 0:
        return torch.zeros((len(detected_bboxes),), device=detected_bboxes.device).long(), None
    # gt_bboxes[:, 2] = gt_bboxes[:, 2] - gt_bboxes[:, 5] * 0.5
    # iou_matrix = get_3d_iou(detected_bboxes, gt_bboxes)

    iou_matrix = BboxOverlaps3D(coordinate='lidar')(detected_bboxes, gt_bboxes)

    # use hungarian algorithm to match bbox
    matched_idx = torch.full((len(detected_bboxes),), -1, device=detected_bboxes.device, dtype=torch.long)
    
    for i in range(len(detected_bboxes)):
        # distances = torch.sum((detected_centers[i].repeat(len(tod3_centers), 1) - tod3_centers) ** 2, dim=1)
        # if torch.min(distances, dim=0)[0] > 3:
        #     matched_idx[i] = -1
        # else:
        #     matched_idx[i] = torch.min(distances, dim=0)[1] # modi2
        # if torch.max(iou_matrix[i], dim=0)[0].item() > 0.2:
        #     matched_idx[i] = torch.max(iou_matrix[i], dim=0)[1]
        # else:
        #     matched_idx[i] = -1
        if torch.max(iou_matrix[i], dim=0)[0].item() > 0.01:
            matched_idx[i] = torch.max(iou_matrix[i], dim=0)[1]
    
    # # for debug
    # detected_bboxes_np = detected_bboxes.detach().cpu().numpy()
    # gt_bboxes_np = gt_bboxes.detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # import time
    # plt.figure()
    # plt.scatter(detected_bboxes_np[:, 0], detected_bboxes_np[:, 1], c='r', label='Detected Centers')
    # plt.scatter(gt_bboxes_np[:, 0], gt_bboxes_np[:, 1], c='b', label='ToD3 Centers')
    # plt.legend()
    # os.makedirs('./output/matched_figure', exist_ok=True)
    # plt.savefig(f'./output/matched_figure/matched_figure_{time.time()}.png', dpi=300)
    # plt.close()

    return matched_idx.view((-1)), iou_matrix

def pop_elem_in_result(task_result:dict, pop_list:list=None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith('query') or k.endswith('query_pos') or k.endswith('embedding'):
            task_result.pop(k)
    
    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result
