from mmcv.parallel import DataContainer
    
def remove_datacontainer(data):
    """Recursively traverse the data structure, removing DataContainer."""
    if isinstance(data, DataContainer):
        return remove_datacontainer(data.data)
    elif isinstance(data, dict):
        return {key: remove_datacontainer(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [remove_datacontainer(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(remove_datacontainer(item) for item in data)
    else:
        return data  # Return other types as is
