"""C 的模型服务层：加载、生成、模型切换、微调对比、性能实验与样例导出。

安装依赖（在项目虚拟环境内）：
    python -m pip install torch pytest psutil matplotlib gradio==5.49.1
测试：python -m pytest tests -q
UI：python -m ui.app
最终实验：python -m ui.model_service --task final --prompt "hello" --device cpu

待接入：
1. TODO(A)：修改 build_model_from_config；模型普通/缓存返回值见 kv_cache.py。
2. TODO(B)：提供 MODEL_SPECS 中的 checkpoint 与训练词表；config 用普通字典保存。
   checkpoint 至少含 model_state_dict、config、training_step、validation_loss。
3. tokenizer.json 是本代码补充的保存约定（原文未指定词表文件格式）：
   B 使用本包 CharTokenizer.save 保存，或在 checkpoint 中增加 tokenizer 字典。
不读取 prompt 临时建词表；必须恢复训练时的 token↔ID 映射。
"""
import argparse
import json
from pathlib import Path
from threading import RLock
import time
import torch

from src.tokenizer.char_tokenizer import CharTokenizer
from src.inference.generate import generate as generate_tokens, get_context_length
from src.optimization.benchmark import seeded, synchronize
from src.optimization.kv_cache_benchmark import compare_kv_cache, run_kv_cache_sweep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# TODO(B)：以下 checkpoint 名称遵循原文；词表路径按 B 实际交付修改。
# 微调沿用原词表时两项可指向同一个 tokenizer.json；若不同，分别指定各自词表。
MODEL_SPECS = {
    "pretrain": {"checkpoint": PROJECT_ROOT / "checkpoints/pretrain_best.pt",
                 "tokenizer": PROJECT_ROOT / "checkpoints/tokenizer.json"},
    "finetune": {"checkpoint": PROJECT_ROOT / "checkpoints/finetune_best.pt",
                 "tokenizer": PROJECT_ROOT / "checkpoints/tokenizer.json"},
}


def build_model_from_config(config: dict):
    """唯一模型构造适配点；由 A 的真实类名和构造函数确定。"""
    # TODO(A)：拿到 A 的代码后，按实际名称取消/修改下面两行并替换 raise：
    # from src.model.gpt import GPT, GPTConfig
    # return GPT(GPTConfig(**config))
    # 如果 A 接收字典，则是 return GPT(config)；不要同时使用两种写法。
    raise NotImplementedError("TODO(A)：请在 ui/model_service.py 的 build_model_from_config 中接入 GPT 类")


class ModelService:
    def __init__(self, model_factory=None, device=None):
        self.model = None
        self.tokenizer = None
        self.config = None
        self.context_length = None
        self.checkpoint_path = None
        self.metadata = {}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("当前服务支持 cpu / cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("当前环境不可用 CUDA，请选择 cpu")
        self.model_factory = model_factory or build_model_from_config
        self._lock = RLock()  # 防止 UI 请求在同一个服务中同时切换模型与推理。

    def load_model(self, checkpoint_path, tokenizer_path=None):
        """先用局部变量完成全部检查，成功后再切换当前模型。"""
        with self._lock:
            path = Path(checkpoint_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"TODO(B)：缺少 checkpoint：{path}")
            # config/metadata 请保存为 dict/int/float 等基本类型。
            # 官方序列化说明：https://docs.pytorch.org/docs/stable/notes/serialization
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            required = {"model_state_dict", "config", "training_step", "validation_loss"}
            if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
                raise ValueError(f"checkpoint 需要字段：{sorted(required)}")
            config = checkpoint["config"]
            if not isinstance(config, dict):
                raise TypeError("TODO(B)：请把 checkpoint['config'] 保存为普通字典")
            embedded = CharTokenizer.from_dict(checkpoint["tokenizer"]) if "tokenizer" in checkpoint else None
            if tokenizer_path is not None and Path(tokenizer_path).is_file():
                tokenizer = CharTokenizer.load(tokenizer_path)
                if embedded is not None and tokenizer.chars != embedded.chars:
                    raise ValueError("外部词表与 checkpoint 内词表的 ID 顺序不一致")
            elif embedded is not None:
                tokenizer = embedded
            else:
                raise FileNotFoundError("TODO(B)：提供训练时的 tokenizer.json 或 checkpoint['tokenizer']")
            if config.get("vocab_size") != tokenizer.vocab_size:
                raise ValueError("config.vocab_size 与训练词表不匹配")
            limit = config.get("context_length", config.get("block_size"))
            if type(limit) is not int or limit < 1:
                raise ValueError("config 需要正整数 context_length（或 block_size）")
            model = self.model_factory(config)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.to(self.device)
            model.eval()
            # 词表大小相同还不够，映射顺序的来源必须由 B 保证；有内嵌词表则上面会比对。
            self.model, self.tokenizer, self.config = model, tokenizer, config
            self.context_length, self.checkpoint_path = limit, path
            self.metadata = {key: checkpoint[key] for key in ("training_step", "validation_loss")}
            return {"checkpoint": str(path), "vocab_size": tokenizer.vocab_size,
                    "context_length": limit, "device": str(self.device), **self.metadata}

    def attach_model(self, model, tokenizer, *, context_length=None):
        """Sprint 2：A 已有随机初始化 GPT、B 尚未交 checkpoint 时使用。"""
        with self._lock:
            limit = get_context_length(model, context_length)
            config = getattr(model, "config", None)
            vocab_size = config.get("vocab_size") if isinstance(config, dict) else getattr(config, "vocab_size", None)
            if vocab_size is not None and vocab_size != tokenizer.vocab_size:
                raise ValueError("模型 vocab_size 与 tokenizer 不一致")
            self.model = model.to(self.device).eval()
            self.tokenizer = tokenizer
            self.context_length = limit
            self.config = {"vocab_size": tokenizer.vocab_size, "context_length": limit}
            self.checkpoint_path = None
            self.metadata = {"source": "attached model; checkpoint not supplied"}

    def encode_prompt(self, prompt):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("请先加载模型和训练词表")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Prompt 不能为空")
        # [T] -> [1,T]；Embedding 接收 torch.long，输入与模型必须在同一设备。
        return torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=self.device)

    def generate(self, prompt, max_new_tokens=100, temperature=1.0, top_k=None,
                 use_kv_cache=False, *, strategy="sample", seed=42):
        """返回生成文本、耗时和 tok/s；计时不包含 encode/decode/模型加载。"""
        with self._lock:
            ids = self.encode_prompt(prompt)
            with seeded(seed, self.device):
                synchronize(self.device)
                start = time.perf_counter()
                output = generate_tokens(self.model, ids, max_new_tokens, temperature,
                    top_k, use_kv_cache, context_length=self.context_length, strategy=strategy)
                synchronize(self.device)
                seconds = time.perf_counter() - start
            new_count = output.size(1) - ids.size(1)
            result = {"text": self.tokenizer.decode(output[0].tolist()),
                      "new_text": self.tokenizer.decode(output[0, ids.size(1):].tolist()),
                      "prompt": prompt, "generated_tokens": new_count,
                      "seconds": seconds, "tokens_per_second": new_count / seconds,
                      "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else None,
                      "settings": {"temperature": temperature, "top_k": top_k,
                                   "strategy": strategy, "seed": seed, "use_kv_cache": use_kv_cache}}
            return result

    def benchmark(self, prompt, max_new_tokens=100, repeats=3, precision="fp32"):
        with self._lock:
            ids = self.encode_prompt(prompt)
            return compare_kv_cache(self.model, ids, max_new_tokens,
                context_length=self.context_length, repeats=repeats, precision=precision,
                model_label=str(self.checkpoint_path))


def save_generation_samples(records, path=PROJECT_ROOT / "outputs/generation/generation_samples.txt"):
    """追加真实生成记录，便于答辩引用；同名 .jsonl 保存机器可读版本。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as readable, path.with_suffix(".jsonl").open("a", encoding="utf-8") as raw:
        for record in records:
            raw.write(json.dumps(record, ensure_ascii=False) + "\n")
            readable.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")
    return str(path)


def compare_finetuning(pretrain, finetune, prompt, **settings):
    """相同 prompt、参数与种子；每个模型使用其训练时的词表。"""
    before = pretrain.generate(prompt, **settings)
    after = finetune.generate(prompt, **settings)
    return {"pretrain": before, "finetune": after}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["generate", "compare", "final"], default="generate")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--lengths", nargs="+", type=int, default=[50, 100, 200, 500])
    args = parser.parse_args()
    services = {}
    names = ["pretrain"] if args.task == "generate" else ["pretrain", "finetune"]
    for name in names:
        services[name] = ModelService(device=args.device)
        spec = MODEL_SPECS[name]
        services[name].load_model(spec["checkpoint"], spec["tokenizer"])
    records = [service.generate(args.prompt, args.max_new_tokens, strategy="greedy")
               for service in services.values()]
    save_generation_samples(records)
    print(json.dumps(records, ensure_ascii=False, indent=2))
    if args.task == "final":
        service = services["pretrain"]
        run_kv_cache_sweep(service.model, service.encode_prompt(args.prompt), args.lengths,
            context_length=service.context_length, model_label=str(service.checkpoint_path),
            output_dir=str(PROJECT_ROOT / "outputs/benchmark"))


if __name__ == "__main__":
    main()
