import gc
import torch
import src.models as local_models


def get_model(model_name: str, config: dict, weights_path: str, device):
    model = getattr(local_models, model_name)(**config)
    checkpoint = torch.load(weights_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def clear_memory(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
