import json
import pickle
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from llava.utils import rank0_print

from llava.constants import (
    DEFAULT_SCENE_START_TOKEN,
    DEFAULT_SCENE_TOKEN,
    DEFAULT_SCENE_END_TOKEN,
    DEFAULT_TRAJ_START_TOKEN,
    DEFAULT_TRACK_TOKEN,
    DEFAULT_TRAJ_END_TOKEN,
    DEFAULT_MAP_START_TOKEN,
    DEFAULT_MAP_TOKEN,
    DEFAULT_MAP_END_TOKEN,
    DEFAULT_TRACK_START_TOKEN,
    DEFAULT_TRACK_END_TOKEN,
    DEFAULT_TRAJ_TOKEN
)

def get_sample_split(nusc: NuScenes, sample_token: str) -> str:
    """
    Determine the split of a sample given its token.
    
    Parameters:
        nusc: An instance of the NuScenes dataset.
        sample_token: The token of the sample (string).
    
    Returns:
        The split the sample belongs to, e.g., 'train', 'val', or 'test'.
    """
    sample_record = nusc.get("sample", sample_token)
    scene_token = sample_record["scene_token"]
    scene_record = nusc.get("scene", scene_token)
    scene_name = scene_record["name"]
    splits = create_splits_scenes()
    for split_name, scene_tokens in splits.items():
        if scene_name in scene_tokens:
            return split_name
    return "unknown"

def process_traj_data(data, split, nusc):

    # id_to_info = process_train_data(input_path)
    converted_entries = []
    
    for sample_token, value in data.items():
        
        if get_sample_split(nusc, sample_token) != split:
            continue

        converted_entry = {
            "qa_id": sample_token+'_trajectory',
            "sample_id": sample_token,
            "conversations": [
                {
                    "from": "human",
                    "value":"",
                },
                {
                    "from": "gpt",
                    "value":"",
                }
            ]
        }
        
        converted_entries.append(converted_entry)
    rank0_print(f"Loaded total {len(converted_entries)} samples")

    return converted_entries

def generate_user_message(data_dict):
        
    """
    Ego-States:
        gt_ego_lcf_feat: [vx, vy, ?, ?, v_yaw (rad/s), ego_length, ego_width, v0 (vy from canbus), Kappa (steering)]
    """
    ego_message = ""
    vx = data_dict['gt_ego_lcf_feat'][0]*0.5
    vy = data_dict['gt_ego_lcf_feat'][1]*0.5
    v_yaw = data_dict['gt_ego_lcf_feat'][4]
    ax = data_dict['gt_ego_his_diff'][-1, 0] - data_dict['gt_ego_his_diff'][-2, 0]
    ay = data_dict['gt_ego_his_diff'][-1, 1] - data_dict['gt_ego_his_diff'][-2, 1]
    cx = data_dict['gt_ego_lcf_feat'][2]
    cy = data_dict['gt_ego_lcf_feat'][3]
    vhead = data_dict['gt_ego_lcf_feat'][7]*0.5
    steeling = data_dict['gt_ego_lcf_feat'][8]
    # ego_message += f"Ego states:"
    ego_message += f"- Velocity (vx,vy): ({vx:.2f},{vy:.2f})"
    ego_message += f" - Heading Angular Velocity (v_yaw): ({v_yaw:.2f})"
    ego_message += f" - Acceleration (ax,ay): ({ax:.2f},{ay:.2f})"
    ego_message += f" - Can Bus: ({cx:.2f},{cy:.2f})"
    ego_message += f" - Heading Speed: ({vhead:.2f})"
    ego_message += f" - Steering: ({steeling:.2f})"

    """
    Historical Trjectory:
        gt_ego_his_trajs: [5, 2] last 2 seconds 
        gt_ego_his_diff: [4, 2] last 2 seconds, differential format, viewed as velocity 
    """
    # his_message = ""
    xh1 = data_dict['gt_ego_his_trajs'][0][0]
    yh1 = data_dict['gt_ego_his_trajs'][0][1]
    xh2 = data_dict['gt_ego_his_trajs'][1][0]
    yh2 = data_dict['gt_ego_his_trajs'][1][1]
    xh3 = data_dict['gt_ego_his_trajs'][2][0]
    yh3 = data_dict['gt_ego_his_trajs'][2][1]
    xh4 = data_dict['gt_ego_his_trajs'][3][0]
    yh4 = data_dict['gt_ego_his_trajs'][3][1]
    # his_message += f"Historical trajectory (last 2 seconds):"
    his_message = f"[({xh1:.2f},{yh1:.2f}),({xh2:.2f},{yh2:.2f}),({xh3:.2f},{yh3:.2f}),({xh4:.2f},{yh4:.2f})]"
    
    """
    Mission goal:
        gt_ego_fut_cmd
    """
    # cmd_message = ""
    cmd_vec = data_dict['gt_ego_fut_cmd']
    right, left, forward = cmd_vec
    if right > 0:
        mission_goal = "turn right"
    elif left > 0:
        mission_goal = "turn left"
    else:
        assert forward > 0
        mission_goal = "keep forward"
    # cmd_message += f"Mission Goal: "
    cmd_message = f"{mission_goal}"
    
    """
    Planning trajectory:
        gt_ego_fut_trajs: [6, 2] last 6 seconds
    """
    x1 = data_dict['gt_ego_fut_trajs'][1][0]
    x2 = data_dict['gt_ego_fut_trajs'][2][0]
    x3 = data_dict['gt_ego_fut_trajs'][3][0]
    x4 = data_dict['gt_ego_fut_trajs'][4][0]
    x5 = data_dict['gt_ego_fut_trajs'][5][0]
    x6 = data_dict['gt_ego_fut_trajs'][6][0]
    y1 = data_dict['gt_ego_fut_trajs'][1][1]
    y2 = data_dict['gt_ego_fut_trajs'][2][1]
    y3 = data_dict['gt_ego_fut_trajs'][3][1]
    y4 = data_dict['gt_ego_fut_trajs'][4][1]
    y5 = data_dict['gt_ego_fut_trajs'][5][1]
    y6 = data_dict['gt_ego_fut_trajs'][6][1]
    
    traj_message = f"[({x1:.2f},{y1:.2f}),({x2:.2f},{y2:.2f}),({x3:.2f},{y3:.2f}),({x4:.2f},{y4:.2f}),({x5:.2f},{y5:.2f}),({x6:.2f},{y6:.2f})]"

    return ego_message, his_message, cmd_message, traj_message


def build_llava_conversation(data_sample, cached_nuscenes_data):

    sample_token = data_sample.get('id', data_sample.get('qa_id')).split('_')[0]
    value = cached_nuscenes_data[sample_token]

    ego_message, his_message, cmd_message, traj_message = generate_user_message(value)
    data_sample['conversations'][0]['value'] = (
        f"Scene information: {DEFAULT_SCENE_START_TOKEN}{DEFAULT_SCENE_TOKEN}{DEFAULT_SCENE_END_TOKEN}\n"
        f"Object-wise tracking information: {DEFAULT_TRACK_START_TOKEN}{DEFAULT_TRACK_TOKEN}{DEFAULT_TRACK_END_TOKEN}\n"
        f"Map information: {DEFAULT_MAP_START_TOKEN}{DEFAULT_MAP_TOKEN}{DEFAULT_MAP_END_TOKEN}\n"
        f"Ego states: {ego_message}\n"
        f"Historical trajectory (last 2 seconds): {his_message}\n"
        f"Mission goal: {cmd_message}\n"
        f"Planning trajectory: {DEFAULT_TRAJ_TOKEN}"
    )
    data_sample['conversations'][1]['value'] = f"{DEFAULT_TRAJ_START_TOKEN}{traj_message}{DEFAULT_TRAJ_END_TOKEN}"
        
    return data_sample
