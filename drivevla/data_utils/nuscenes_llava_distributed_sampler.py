from typing import Dict, Optional, List
from torch.utils.data.distributed import DistributedSampler
import torch
from tqdm import tqdm

class ContinuousSceneDistributedSampler(DistributedSampler):
    """A custom sampler that ensures samples from the same scene are assigned to the same GPU
    and maintains scene continuity within each GPU's partition.
    
    This sampler has the following features:
    1. Samples from the same scene are always assigned to the same GPU
    2. Each GPU gets a continuous block of scenes
    3. Scene blocks are distributed to balance the load across GPUs
    4. Within each GPU, the order of samples in a scene is maintained
    5. Optional shuffling only happens at the scene level, not within scenes
    """
    
    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas,
                        rank=rank, shuffle=shuffle, seed=seed,
                        drop_last=drop_last)
        
        # Group indices by scene
        self.scene_groups = self._group_by_scene(dataset)
        
        # Calculate scene assignments to each GPU
        self.scene_assignments = self._assign_scenes_to_gpus()
        
        # Get indices for this GPU's assigned scenes
        self.indices = self._get_rank_indices()
        
        # Update num_samples based on actual indices assigned
        self.num_samples = len(self.indices)
        
        # Print distribution info
        if self.rank == 0:
            self._print_distribution_info()
        
    def _group_by_scene(self, dataset) -> Dict[str, List[int]]:
        """Group dataset indices by scene token."""
        scene_groups = {}
        
        for idx in range(len(dataset)):
            # Get scene token of the sample
            scene_token = dataset.data_infos[idx]['scene_token']
            
            if scene_token not in scene_groups:
                scene_groups[scene_token] = []
            scene_groups[scene_token].append(idx)
            
        return scene_groups
    
    def _assign_scenes_to_gpus(self) -> Dict[int, List[str]]:
        """Assign scenes to GPUs in a continuous manner."""
        # Get all scene token lists
        scene_tokens = list(self.scene_groups.keys())
        
        # Initialize GPU assignments dictionary
        gpu_assignments = {i: [] for i in range(self.num_replicas)}
        
        # Calculate the number of scenes per GPU (except the last one)
        base_scenes_per_gpu = len(scene_tokens) // self.num_replicas
        
        # Assign scenes to each GPU in order
        for gpu_id in range(self.num_replicas - 1):
            start_idx = gpu_id * base_scenes_per_gpu
            end_idx = start_idx + base_scenes_per_gpu
            gpu_assignments[gpu_id] = scene_tokens[start_idx:end_idx]
        
        # The last GPU gets all remaining scenes
        last_gpu_id = self.num_replicas - 1
        start_idx = (self.num_replicas - 1) * base_scenes_per_gpu
        gpu_assignments[last_gpu_id] = scene_tokens[start_idx:]
            
        return gpu_assignments
    
    def _get_rank_indices(self) -> List[int]:
        """Get the indices assigned to current rank/GPU."""
        indices = []
        for scene_token in self.scene_assignments[self.rank]:
            indices.extend(self.scene_groups[scene_token])
        return indices
    
    def _print_distribution_info(self):
        """Print information about the data distribution across GPUs."""
        print("\nScene Distribution Information:")
        print(f"Total number of scenes: {len(self.scene_groups)}")
        print(f"Number of GPUs: {self.num_replicas}")
        
        # Calculate statistics for each GPU
        gpu_stats = []
        for gpu_id in range(self.num_replicas):
            scenes = self.scene_assignments[gpu_id]
            total_samples = sum(len(self.scene_groups[scene]) for scene in scenes)
            gpu_stats.append({
                'gpu_id': gpu_id,
                'num_scenes': len(scenes),
                'num_samples': total_samples
            })
        
        # Print statistics
        print("\nPer-GPU Statistics:")
        for stats in gpu_stats:
            print(f"GPU {stats['gpu_id']}: {stats['num_scenes']} scenes, {stats['num_samples']} samples")
        
        # Calculate and print load balance metrics
        sample_loads = [stats['num_samples'] for stats in gpu_stats]
        min_load = min(sample_loads)
        max_load = max(sample_loads)
        avg_load = sum(sample_loads) / len(sample_loads)
        imbalance = (max_load - min_load) / avg_load * 100 if avg_load > 0 else 0
        
        print(f"\nLoad Balance Metrics:")
        print(f"Min samples per GPU: {min_load}")
        print(f"Max samples per GPU: {max_load}")
        print(f"Average samples per GPU: {avg_load:.2f}")
        print(f"Load imbalance: {imbalance:.2f}%\n")
    
    def __iter__(self):
        if self.shuffle:
            # Shuffle scenes assigned to this GPU
            scene_indices = list(range(len(self.scene_assignments[self.rank])))
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            scene_perm = torch.randperm(len(scene_indices), generator=g).tolist()
            shuffled_scenes = [self.scene_assignments[self.rank][i] for i in scene_perm]
            
            # Get indices for shuffled scenes while maintaining continuity within scenes
            indices = []
            for scene_token in shuffled_scenes:
                indices.extend(self.scene_groups[scene_token])
        else:
            indices = self.indices
            
        return iter(indices)
    
    def __len__(self) -> int:
        return self.num_samples