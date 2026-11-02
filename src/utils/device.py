import torch


def get_device(name="auto"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("仅支持 cpu / cuda / auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请使用 --device cpu")
    return device
