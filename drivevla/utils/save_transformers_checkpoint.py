import os
import torch
from llava.model.multimodal_encoder.uniad_track_map import UniadTrackMapVisionTower

def save_transformers_checkpoint(model, output_dir):
    """Convert UniadTrackMapVisionTower model to transformers checkpoint format.
    
    Args:
        model: The UniadTrackMapVisionTower model instance
        output_dir: Output directory to save the converted checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. save config
    config_dict = model.config.to_dict()
    model.config.save_pretrained(output_dir)
    
    # 2. save model weights
    state_dict = model.state_dict()
    
    # 3. save model weights
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    
    print(f"Checkpoint saved to {output_dir}")

def main():
    vision_tower = UniadTrackMapVisionTower(vision_tower="uniad_track_map", vision_tower_cfg=None)
    save_transformers_checkpoint(vision_tower.vision_tower, "checkpoints/uniad_track_map")

if __name__ == '__main__':
    main()
