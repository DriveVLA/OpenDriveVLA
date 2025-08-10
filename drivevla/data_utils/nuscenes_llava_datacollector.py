import torch
import transformers
from dataclasses import dataclass
from typing import Dict, Sequence
from mmcv.parallel import collate
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_SCENE_TOKEN, DEFAULT_TRACK_TOKEN, DEFAULT_MAP_TOKEN, IMAGE_TOKEN_INDEX
from drivevla.utils.remove_mmlab_datacontainer import remove_datacontainer

@dataclass
class DataCollatorForLLaVANuScenesDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    llava_train_mode: bool = False
    llava_test_mode: bool = False

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if self.llava_train_mode:
            return self._train_call(instances)
        elif self.llava_test_mode:
            return self._test_call(instances)
        else:
            raise ValueError("Invalid mode, please specify either llava_train_mode or llava_test_mode to be True")

    def _train_call(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        assert len(instances) == 1, "Currently only one instance (batch_size=1) is supported for UniADTrackMapVisionTower inference"
        if "uniad_pth" in instances[0]:
            batch["uniad_pth"] = instances[0]["uniad_pth"]

        if "uniad_data" in instances[0]:
            instance = instances[0]

            uniad_data = [instance["uniad_data"]]
            uniad_data = collate(uniad_data)

            # remove DataContainer to avoid GPU memory leak
            uniad_data = remove_datacontainer(uniad_data)
            uniad_data['img_metas'] = uniad_data['img_metas'][0]
            uniad_data['img'] = uniad_data['img'][0]
            uniad_data['l2g_r_mat'] = uniad_data['l2g_r_mat'][0]
            uniad_data['l2g_t'] = uniad_data['l2g_t'][0]
            uniad_data['gt_bboxes_3d'] = uniad_data['gt_bboxes_3d'][0]
            uniad_data['gt_labels_3d'] = uniad_data['gt_labels_3d'][0]
            uniad_data['gt_past_traj'] = uniad_data['gt_past_traj'][0]
            uniad_data['gt_past_traj_mask'] = uniad_data['gt_past_traj_mask'][0]
            uniad_data['gt_inds'] = uniad_data['gt_inds'][0]
            uniad_data['gt_sdc_bbox'] = uniad_data['gt_sdc_bbox'][0]
            uniad_data['gt_sdc_label'] = uniad_data['gt_sdc_label'][0]

            uniad_data['timestamp'] = uniad_data['timestamp'][0]
            timestamp_0 = uniad_data['timestamp'][0][0]
            for i in range(len(uniad_data['timestamp'][0])):
                uniad_data['timestamp'][0][i] = uniad_data['timestamp'][0][i] - timestamp_0
            
            batch["uniad_data"] = uniad_data

        if "qa_instance_ind" in instances[0]:
            batch["qa_instance_ind"] = instances[0]["qa_instance_ind"]

        return batch
    
    def _test_call(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert len(instances) == 1, "Currently only one instance (batch_size=1) is supported for UniADTrackMapVisionTower inference"

        instance = instances[0]

        if "uniad_data" in instance:
            uniad_data = [instance["uniad_data"]]
            uniad_data = collate(uniad_data)

            # remove DataContainer to avoid GPU memory leak
            uniad_data = remove_datacontainer(uniad_data)
            uniad_data['img_metas'] = uniad_data['img_metas'][0]
            uniad_data['img'] = uniad_data['img'][0]
            
            instance["uniad_data"] = uniad_data

        if "qa_instance_ind" in instance:
            instance["qa_instance_ind"] = instance["qa_instance_ind"]

        return instance
