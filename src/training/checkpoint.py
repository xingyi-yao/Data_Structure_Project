from pathlib import Path
import os
import tempfile
import torch
from config import normalize_model_config
from src.model.gpt import GPT
from src.tokenizer import tokenizer_from_dict


def save_checkpoint(path, model, tokenizer, optimizer=None, *, training_step=0,
                    validation_loss=float("inf"), best_validation_loss=None,
                    history=None, training_config=None, data_signature=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(model_state_dict=model.state_dict(), config=dict(model.config),
                 tokenizer=tokenizer.to_dict(), training_step=training_step,
                 validation_loss=validation_loss,
                 best_validation_loss=validation_loss if best_validation_loss is None else best_validation_loss,
                 history=history or [], training_config=training_config or {},
                 data_signature=data_signature, torch_rng_state=torch.get_rng_state())
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "config", "tokenizer", "training_step", "validation_loss"}
    if not isinstance(state, dict) or not required.issubset(state):
        raise ValueError(f"checkpoint 缺少必要字段：{sorted(required)}")
    tokenizer = tokenizer_from_dict(state["tokenizer"])
    config = normalize_model_config(state["config"])
    if config["vocab_size"] != tokenizer.vocab_size:
        raise ValueError("模型与词表不匹配")
    model = GPT(**config)
    model.load_state_dict(state["model_state_dict"], strict=True)
    return model.to(device), tokenizer, state
