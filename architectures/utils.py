import os
import torch

def get_saved_model(architecture, weight_path):
    """
    @param weight_path: points to path where weights are stored to load model
    @returns: Trained model
    @throws: Exception if no weights stored at weight_path
    """
    if os.path.isfile(weight_path):
        model = architecture().cpu()
        model.load_state_dict(torch.load(weight_path))
        model.eval()
    else:
        raise Exception(f'No such file {weight_path}.')