import json
import re
import copy
import torch
import numpy as np
from typing import Dict
import math
import random
import yaml
import os
import pickle

from llava.train.train import preprocess
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token

import transformers

from mmdet.datasets.pipelines import to_tensor

from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset

from llava.train.train import DataArguments
from llava.utils import rank0_print
from drivevla.data_utils.build_llava_conversation import build_llava_conversation, process_traj_data

class LLaVANuScenesDataset(NuScenesE2EDataset):
    r"""NuScenes E2E Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 NuScenesE2EDataset_config: dict,
                 device: torch.device = "cpu",
                 llava_train_mode: bool = False,
                 llava_test_mode: bool = False,
                 use_uniad_pth: bool = False,
                 in_nuscenes_order: bool = True,
                 *args, 
                 **kwargs):
        NuScenesE2EDataset_config.pop('type')
        super().__init__(*args, **NuScenesE2EDataset_config, **kwargs)
        
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.device = device
        self.llava_train_mode = llava_train_mode
        self.llava_test_mode = llava_test_mode
        self.use_uniad_pth = use_uniad_pth
        self.in_nuscenes_order = in_nuscenes_order
        self.cached_nuscenes_data = pickle.load(open('data/nuscenes/cached_nuscenes_info.pkl', 'rb'))
        if self.use_uniad_pth:
            self.in_nuscenes_order = False
        if self.test_mode == True:
            self.ins_inds_add_1_in_pipeline = NuScenesE2EDataset_config['pipeline'][3]['ins_inds_add_1']
            self.nusc_split = 'val'
        else:
            self.ins_inds_add_1_in_pipeline = NuScenesE2EDataset_config['pipeline'][2]['ins_inds_add_1']
            self.nusc_split = 'train'

        self.list_data_dict = []
        if self.data_args.data_path is None:
            # online generation
            list_data_dict = process_traj_data(self.cached_nuscenes_data, self.nusc_split, self.nusc)
        else:
            list_data_dict = self._load_conversation_data()
        
        self._reorder_data_dict(list_data_dict)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

        print(f"llava_train_mode: {self.llava_train_mode}")
        print(f"llava_test_mode: {self.llava_test_mode}")
        print(f"in_nuscenes_order: {self.in_nuscenes_order}")
        print(f"use_uniad_pth: {self.use_uniad_pth}")

    def _load_conversation_data(self):
        data_path = self.data_args.data_path
        list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            self.data_args.dataset_paths = []
            for file_name in file_names:
                self.data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                self.data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    list_data_dict.extend(cur_data_dict)
        else:
            self.data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                list_data_dict.extend(cur_data_dict)
                
        rank0_print(f"Loaded total {len(list_data_dict)} samples from {data_path}")

        return list_data_dict
    
    def _reorder_data_dict(self, list_data_dict):
        sample_token_to_idx = {}
        if self.in_nuscenes_order:
            for i, data_dict in enumerate(list_data_dict):
                sample_token = data_dict.get('id', data_dict.get('qa_id')).split('_')[0]
                sample_token_to_idx[sample_token] = i

            # Reorder the list_data_dict to match the order of Nuscenes data_infos
            for i in range(len(self.data_infos)):
                sample_token = self.data_infos[i]['token']
                try:
                    self.list_data_dict.append(list_data_dict[sample_token_to_idx[sample_token]])
                except:
                    self.list_data_dict.append({"id": sample_token, "conversations": [{"role": "user", "value": "This sample does not have a conversation."}]})
                    print(f"!!! sample_token '{sample_token}' not found in <sample_token_to_idx>, meaning this sample is not in the conversation data. Please check the conversation data.")
        else:
            self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, idx):
        """Get item according to the given index.
        """
        if self.in_nuscenes_order:
            if self.test_mode == True:
                uniad_data_dict = self._get_uniad_test_data(idx)
            else:
                uniad_data_dict = self._get_uniad_train_data(idx)

        if self.llava_train_mode:  # Training mode
            llava_data_dict = self._get_llava_train_data(idx)
        elif self.llava_test_mode:  # Testing mode
            llava_data_dict = self._get_llava_test_data(idx)
        else:
            raise ValueError("Invalid mode, please specify either llava_train_mode or llava_test_mode to be True")

        uniad_pth_dict = {}
        if self.use_uniad_pth:
            uniad_pth_dict = self._get_uniad_pth_data(idx)

        if self.in_nuscenes_order:
            return uniad_data_dict | llava_data_dict | uniad_pth_dict
        else:
            return llava_data_dict | uniad_pth_dict

    def _get_uniad_train_data(self, idx):
        """Get uniad data from infos according to the given index.
        """
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            else:
                break
        uniad_data_dict = {"uniad_data": data}
        for i in range(len(uniad_data_dict['uniad_data']['img_metas'])):
            uniad_data_dict['uniad_data']['img_metas'].data[i].pop('box_type_3d')
        return uniad_data_dict

    def _get_uniad_test_data(self, idx):
        """Get uniad data from infos according to the given index.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_labels_3d: gt_labels of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame(list)
        """

        input_dict = self.get_data_info(idx)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        uniad_data = {}
        for key, value in example.items():
            if 'l2g' in key:
                if isinstance(value[0], np.float32):
                    uniad_data[key] = to_tensor(float(value[0]))
                else:
                    uniad_data[key] = to_tensor(value[0])
            else:
                uniad_data[key] = value
        uniad_data_dict = {"uniad_data": uniad_data}
        uniad_data_dict['uniad_data']['img_metas'][0]._data.pop('box_type_3d')
        return uniad_data_dict

    def _get_llava_train_data(self, idx):
        source = self.list_data_dict[idx]
        source = build_llava_conversation(source, self.cached_nuscenes_data)
        sources = [source]
        assert len(sources) == 1

        conv = copy.deepcopy([e["conversations"] for e in sources])

        llava_data_dict = preprocess(conv, self.tokenizer, has_image=True)  # <image> -> -200

        llava_data_dict = dict(input_ids=llava_data_dict["input_ids"][0], labels=llava_data_dict["labels"][0])

        llava_data_dict["id"] = source.get("id", source.get("qa_id", idx))

        if "instance_token" in source:
            instance_token = source["instance_token"]
            instance_ind = self.nusc.getind('instance', instance_token)
            if self.ins_inds_add_1_in_pipeline:
                instance_ind += 1
            llava_data_dict["qa_instance_ind"] = instance_ind

        return llava_data_dict

    def _get_llava_test_data(self, idx):
        data = self.list_data_dict[idx]
        data = build_llava_conversation(data, self.cached_nuscenes_data)

        id = data.get("id", data.get("qa_id", idx))
        question = data['conversations'][0]['value']

        # Construct conversation context
        conv = conv_templates["qwen_planning_oriented_vlm"].copy()
        conv.clear_conversation()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        # Convert text to IDs
        input_ids = tokenizer_uniad_token(
            prompt_question, self.tokenizer, return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        llava_data_dict = {
                     "id": id, 
                     "question":question,
                     "input_ids": input_ids}
        
        if "instance_token" in data:
            instance_token = data["instance_token"]
            instance_ind = self.nusc.getind('instance', instance_token)
            if self.ins_inds_add_1_in_pipeline:
                instance_ind += 1
            llava_data_dict["qa_instance_ind"] = instance_ind

        return llava_data_dict
    
    def _get_uniad_pth_data(self, idx):
        data = self.list_data_dict[idx]

        # Load and process uniad_pth data
        if os.path.exists(data['uniad_pth']):
            uniad_pth = torch.load(data['uniad_pth'], map_location=self.device)
        else:
            if self.llava_train_mode:
                os.makedirs('data/uniad_results_for_vlm/train/', exist_ok=True)
                
                print(f"\nFetching {data['sample_id']}.pth from remote server...")

                os.system(f"rsync -a --partial --append-verify --info=progress2 --human-readable ge86wob2@login.ai.lrz.de:/dss/dssfs04/lwp-dss-0002/pn39vu/pn39vu-dss-0001/xuyuan/workspace/repos/drivevlms.worktrees/planning-oriented/llava-next/uniad/data/uniad_results_for_vlm/train/{data['sample_id']}.pth data/uniad_results_for_vlm/train/")
                
                uniad_pth = torch.load(data['uniad_pth'], map_location=self.device)
                
            elif self.llava_test_mode:
                os.makedirs('data/uniad_results_for_vlm/val/', exist_ok=True)
                
                print(f"\nFetching {data['sample_id']}.pth from remote server...")

                os.system(f"rsync -a --partial --append-verify --info=progress2 --human-readable ge86wob2@login.ai.lrz.de:/dss/dssfs04/lwp-dss-0002/pn39vu/pn39vu-dss-0001/xuyuan/workspace/repos/drivevlms.worktrees/planning-oriented/llava-next/uniad/data/uniad_results_for_vlm/val/{data['sample_id']}.pth data/uniad_results_for_vlm/val/")

                uniad_pth = torch.load(data['uniad_pth'], map_location=self.device)

        uniad_pth_dict = {"uniad_pth": uniad_pth}
        return uniad_pth_dict
    
    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            length_list.append(cur_len)
        return length_list