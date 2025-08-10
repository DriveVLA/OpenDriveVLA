import argparse
import json
import os
import sys

from utils.trajectory_utils import retrieve_traj, check_traj
from eval_share.evaluation import planning_evaluation

def evaluate_planning_oriented_vlm(output_path):
    '''
    Convert the planning results from conversations to pred_trajs_dict and pred_trajs_multi_modal_dict
    '''
    out_dir = os.path.dirname(output_path)
    
    pred_trajs_dict = {}
    pred_trajs_multi_modal_dict = {}

    with open(output_path, 'r') as f:
        for line in f:
            conv_result = json.loads(line.strip())

            # retrieve multi-modal planning trajectories
            traj_multi_modal = []
            for answer in conv_result['answer']:
                traj = retrieve_traj(answer)
                check_traj(traj)
                traj_multi_modal.append(traj)

            # TODO: pick the best planning trajectory
            pred_trajs_dict[conv_result['id']] = [traj_multi_modal[0]]
            pred_trajs_multi_modal_dict[conv_result['id']] = traj_multi_modal

    # Save pred_trajs_dict to a json file
    with open(os.path.join(out_dir, "pred_trajs_dict.json"), 'w') as f:
        json.dump(pred_trajs_dict, f, indent=2)
    print(f"Saving pred_trajs_dict to {os.path.join(out_dir, 'pred_trajs_dict.json')}")

    # Save planning_trajectory_multi_modal to a json file
    with open(os.path.join(out_dir, "pred_trajs_multi_modal_dict.json"), 'w') as f:
        json.dump(pred_trajs_multi_modal_dict, f, indent=2)
    print(f"Saving pred_trajs_multi_modal to {os.path.join(out_dir, 'pred_trajs_multi_modal_dict.json')}")

    # Setup logging to save the evaluation results to a file
    log_path = os.path.join(out_dir, "eval_results.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"Saving evaluation results to {log_path}")
    sys.stdout = open(log_path, 'w')

    """
    # TODO: check only_vehicle=False or True
    This evaluation function will report both the STP-3 metric (avg over avg) and the UniAD metric. 
    Since UniAD only considers the vehicle category when generating ground truth occupancy, while ST-P3 considers both the vehicle and pedestrian categories.
    If you want to report the STP-3 metric, please set only_vehicle=False.
    if you want to report the UniAD metric, please set only_vehicle=True.
    """
    planning_evaluation(pred_trajs_dict, subset=None, only_vehicle=True)
    
    # Restore stdout and print log contents
    sys.stdout.close()
    sys.stdout = sys.__stdout__
    
    # Read and print log contents
    with open(log_path, 'r') as f:
        log_content = f.read()
        print(log_content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    evaluate_planning_oriented_vlm(args.output)
    print(f">>> Evaluation results saved to {os.path.dirname(args.output)}")

if __name__ == "__main__":
    main()