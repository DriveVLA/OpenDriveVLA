import torch.nn as nn

def find_meta_tensors(model: nn.Module, prefix=''):
    """Find meta tensors in the model
    
    Args:
        model: PyTorch model
        prefix: prefix for hierarchical display
    """
    meta_tensors = []
    
    # Iterate through all parameters of the model
    for name, param in model.named_parameters():
        if param.device.type == 'meta':
            meta_tensors.append(f"{prefix}{name}")
            
    # Iterate through all buffers of the model (e.g. BatchNorm's running_mean)
    for name, buffer in model.named_buffers():
        if buffer.device.type == 'meta':
            meta_tensors.append(f"{prefix}{name}")
    
    # Recursively check all submodules
    for name, child in model.named_children():
        child_prefix = f"{prefix}{name}."
        meta_tensors.extend(find_meta_tensors(child, child_prefix))
    
    return meta_tensors

def print_meta_tensors(model: nn.Module):
    """Print all meta tensors in the model"""
    meta_tensors = find_meta_tensors(model)
    if meta_tensors:
        print(f">>> Found {len(meta_tensors)} meta tensors")
    else:
        print(">>> No meta tensors found")