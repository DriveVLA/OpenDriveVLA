import torch

def move_data_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_data_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_data_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_data_to_device(item, device) for item in data)
    return data

def change_tensor_to_float16(data):
    if isinstance(data, torch.Tensor):
        if data.dtype in [torch.float32, torch.float64]:
            return data.to(dtype=torch.float16)
        return data
    elif isinstance(data, dict):
        return {key: change_tensor_to_float16(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [change_tensor_to_float16(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(change_tensor_to_float16(item) for item in data)
    return data

def change_tensor_to_float32(data):
    if isinstance(data, torch.Tensor):
        if data.dtype in [torch.float16, torch.bfloat16]:
            return data.to(dtype=torch.float32)
        return data
    elif isinstance(data, dict):
        return {key: change_tensor_to_float32(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [change_tensor_to_float32(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(change_tensor_to_float32(item) for item in data)
    return data

def check_non_cpu_tensors(obj, path=""):
    """
    Recursively check and show tensors that are not on CPU
    Args:
        obj: The object to check
        path: Current path in the object hierarchy
    Returns:
        List of (path, device, shape) for non-CPU tensors
    """
    results = []
    
    if isinstance(obj, torch.Tensor):
        # Check if not on CPU
        if obj.device.type != 'cpu':
            results.append({
                'path': path,
                'device': str(obj.device),
                'shape': list(obj.shape),
                'dtype': str(obj.dtype)
            })
    elif isinstance(obj, dict):
        for key, value in obj.items():
            results.extend(check_non_cpu_tensors(value, f"{path}.{key}" if path else key))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            results.extend(check_non_cpu_tensors(item, f"{path}[{i}]"))
    
    return results

def print_non_cpu_tensor_info(data):
    """
    Print non-CPU tensor information in a formatted way
    """
    results = check_non_cpu_tensors(data)
    if not results:
        print("\nAll tensors are on CPU.")
        return
        
    print("\n=== Non-CPU Tensor Information ===")
    for info in results:
        print(f"\nPath: {info['path']}")
        print(f"Device: {info['device']}")
        print(f"Shape: {info['shape']}")
        print(f"Dtype: {info['dtype']}")
    print("\n===================================")