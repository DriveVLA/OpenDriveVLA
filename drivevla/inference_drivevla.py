import argparse
import torch
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
import time
import json
import os
from PIL import Image
from tqdm import tqdm
import deepspeed

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

from mmengine import Config
from llava.train.train import DataArguments

from data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset

from utils.tensor_utils import move_data_to_device
from data_utils.nuscenes_llava_distributed_sampler import ContinuousSceneDistributedSampler

def read_processed_ids(output_file):
    processed_ids = set()
    try:
        with open(output_file, 'r') as f:
            for line in f:
                conv_result = json.loads(line.strip())
                processed_ids.add(conv_result['id'])
    except FileNotFoundError:
        # Handle the case where the file does not exist
        pass
    return processed_ids

def load_image(cam_path):
    return Image.open(cam_path).convert('RGB')

def load_model_with_deepspeed(args, device):
    """
    Load model with DeepSpeed configuration for inference
    """
    disable_torch_init()

    # Load model
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": args.attn_implementation
    }
    
    overwrite_config = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}

    llava_model_args["overwrite_config"] = overwrite_config

    # Load the model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path,
        model_base=None,
        model_name="llava_qwen",
        device_map=device,
        **llava_model_args
    )

    # DeepSpeed inference configuration
    ds_config = {
        "fp16": {"enabled": args.fp16},
        "bf16": {"enabled": args.bf16},
        "zero_optimization": {
            "stage": 0
        },
        "train_micro_batch_size_per_gpu": args.batch_size,
        "wall_clock_breakdown": False,
        "inference_mode": True
    }

    # Initialize DeepSpeed engine for inference
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=[]
    )

    return tokenizer, model_engine, image_processor, context_len

def inference_data(data, model_engine, tokenizer, args):
    """
    Process the data_item to create inputs for the model, generate answers, and return results.
    """
    id = data["id"]
    question = data["question"]
    input_ids = data["input_ids"]
    uniad_data = data.get("uniad_data", None)
    uniad_pth = data.get("uniad_pth", None)
    qa_instance_ind = data.get("qa_instance_ind", None)
    
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.bf16 else torch.float16):
            cont = model_engine.generate(
                input_ids,
                uniad_data=uniad_data,
                uniad_pth=uniad_pth,
                qa_instance_ind=qa_instance_ind,
                do_sample=False,
                temperature=0,
                max_new_tokens=512,
                num_beams=1,
            )

            # # multi-modal trajectory generation
            # cont = model_engine.generate(
            #     input_ids,
            #     uniad_data=uniad_data,
            #     uniad_pth=uniad_pth,
            #     qa_instance_ind=qa_instance_ind,
            #     do_sample=True,
            #     temperature=0.1,
            #     num_return_sequences=4,
            #     max_new_tokens=512,
            #     num_beams=1,
            # )

    answer = tokenizer.batch_decode(cont, skip_special_tokens=True)
    result = {
        'id': id,
        'question': question,
        'answer': answer
    }
    return result

def inference_planning_oriented_vlm(args):
    # Initialize DDP
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    # Load data
    rank = dist.get_rank() if args.local_rank != -1 else 0
    world_size = dist.get_world_size() if args.local_rank != -1 else 1

    # Create output directory on rank 0 and synchronize path across all ranks
    if args.local_rank != -1:
        if rank == 0:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Broadcast the output path from rank 0 to all other ranks
        if world_size > 1:
            if rank == 0:
                output_path = args.output
            else:
                output_path = None
            output_path = [output_path]
            dist.broadcast_object_list(output_path, src=0)
            if rank != 0:
                args.output = output_path[0]
        
        # Make sure all ranks have the output directory
        if rank != 0:
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
        dist.barrier()
    else:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = f"output/{args.model_base}/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        args.output = f"{output_dir}/planning_conversations_val.json"

    device = torch.device(args.local_rank if args.local_rank != -1 else 'cuda')
    
    # Load model with DeepSpeed
    tokenizer, model_engine, image_processor, context_len = load_model_with_deepspeed(args, device)
    model_engine.eval()

    uniad_cfg: Config = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
    data_args = DataArguments(
        data_path=args.data,
        lazy_preprocess=True,
        frames_upbound=32,
    )

    test_dataset = LLaVANuScenesDataset(tokenizer, data_args, uniad_cfg.data.test, llava_test_mode=True, use_uniad_pth=args.use_uniad_pth)
    # test_dataset = LLaVANuScenesDataset(tokenizer, data_args, uniad_cfg.data.test_llava_with_track_gt, llava_test_mode=True, use_uniad_pth=args.use_uniad_pth)
    
    # Initialize DDP sampler
    if args.local_rank != -1 and args.world_size > 1:
        if args.use_uniad_pth:  # in_nuscenes_order = False
            sampler = DistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False
            )
            print("Using DistributedSampler")
        else:  # in_nuscenes_order = True
            sampler = ContinuousSceneDistributedSampler(
                test_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False
            )
            print("Using ContinuousSceneDistributedSampler")
    else:
        sampler = None
        print("Not using DDP sampler")
        
    data_collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)

    # Enable torch.backends optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Prefetch factor for dataloader
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
        sampler=sampler,
        shuffle=False
    )

    # Create output file with rank suffix for each GPU
    rank_output = args.output.replace('.json', f'_rank{rank}.json')

    # Create a new file or clear existing one
    if os.path.exists(rank_output):
        os.remove(rank_output)
    
    # Print dataset distribution info
    if rank == 0:
        print(f"Total dataset size: {len(test_dataset)}")
        print(f"Number of GPUs (world_size): {world_size}")
        print(f"Approximate samples per GPU: {len(test_dataset) // world_size}")
        print(f"Output directory: {os.path.dirname(args.output)}")
    
    tqdm_bar = tqdm(dataloader, ncols=80, disable=rank != 0)
    
    # Set sampler epoch to ensure proper sharding
    if sampler is not None:
        sampler.set_epoch(0)
    
    for data in tqdm_bar:
        data = move_data_to_device(data, device)
        
        start_time = time.time()
        result = inference_data(data, model_engine, tokenizer, args)
        inference_time = time.time() - start_time
        
        if rank == 0:
            tqdm_bar.set_postfix_str(f'Inference: {inference_time:.2f}s')
        
        # Each GPU writes to its own file
        with open(rank_output, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Wait for all processes to finish
    if args.local_rank != -1:
        dist.barrier()

    # Merge results from all ranks if this is the main process
    if rank == 0 and args.local_rank != -1:
        merge_results(args)

def merge_results(args):
    """Merge results from all ranks into a single file"""
    world_size = dist.get_world_size()
    all_results = []
    
    # Read results from each rank
    for rank in range(world_size):
        rank_output = args.output.replace('.json', f'_rank{rank}.json')
        if os.path.exists(rank_output):
            with open(rank_output, 'r', encoding='utf-8') as f:
                for line in f:
                    all_results.append(json.loads(line.strip()))
            
    # Write merged results to a new file to avoid any conflicts
    final_output = args.output
    with open(final_output, 'w', encoding='utf-8') as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
    # Clean up rank files
    # for rank in range(world_size):
    #     rank_output = args.output.replace('.json', f'_rank{rank}.json')
    #     if os.path.exists(rank_output):
    #         os.remove(rank_output)
    
    print(f"Successfully merged results from {world_size} workers into {args.output}")

def main():
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)

    model_path_default = "checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=model_path_default)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--resume-from-output", type=str, default=None)  # path to the output planning_conversations_val.json
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use-uniad-pth", action="store_true", help="Use uniad pth for inference")
    parser.add_argument("--attn-implementation", type=str, default="sdpa",
                      choices=["sdpa", "flash_attention_2", "eager"],
                      help="Attention implementation to use")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    parser.add_argument("--zero-stage", type=int, default=2, choices=[0, 1, 2, 3],
                        help="ZeRO optimization stage")
    
    # DDP related arguments
    parser.add_argument("--local_rank", type=int, default=int(os.getenv('LOCAL_RANK', -1)),
                        help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=int(os.getenv('WORLD_SIZE', -1)),
                        help="World size for DDP")
    parser.add_argument("--dist-url", default="env://", type=str,
                        help="URL used to set up distributed training")

    args = parser.parse_args()

    # Set up distributed training environment
    if args.local_rank != -1:
        if 'RANK' not in os.environ:
            os.environ['RANK'] = str(args.local_rank)
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = str(args.world_size)
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'

    if args.resume_from_output is not None:
        args.output = args.resume_from_output

    if args.model_base is None:
        # Remove 'checkpoints/' and everything before it, then replace remaining '/' with '_'
        args.model_base = args.model_path.split('checkpoints/')[-1].replace('/', '_')
        if args.output is None:
            args.output = f"output/{args.model_base}/{time.strftime('%Y%m%d_%H%M%S')}/plan_conv.json"

    inference_planning_oriented_vlm(args)
    
    # Only print completion message on main process
    if args.local_rank in [-1, 0]:
        print(f">>> Inference results saved to {args.output}")

if __name__ == "__main__":
    main()