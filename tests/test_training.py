"""训练保存格式与真实模型服务互通；本次未执行。"""
from pathlib import Path
import tempfile
import unittest
import torch
from src.model.gpt import GPT
from src.tokenizer import CharTokenizer
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.dataset import TextDataset
from src.training.evaluate import evaluate
from ui.model_service import ModelService


class TestTrainingIntegration(unittest.TestCase):
    def test_checkpoint_loads_in_ui_and_resumes_optimizer(self):
        tokenizer = CharTokenizer("abc")
        model = GPT(3, context_length=4, n_embd=8, n_head=2, n_layer=1, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters())
        x = torch.tensor([[0, 1, 2, 0]])
        _, loss = model(x, torch.tensor([[1, 2, 0, 1]]))
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, tokenizer, optimizer, training_step=1, validation_loss=1.2)
            restored, vocabulary, state = load_checkpoint(path)
            self.assertEqual(vocabulary.to_dict(), tokenizer.to_dict())
            torch.testing.assert_close(restored(x)[0], model(x)[0])
            new_optimizer = torch.optim.AdamW(restored.parameters())
            new_optimizer.load_state_dict(state["optimizer_state_dict"])
            self.assertTrue(new_optimizer.state)
            service = ModelService(device="cpu")
            service.load_model(path)
            self.assertEqual(service.generate("ab", 2, use_kv_cache=True)["generated_tokens"], 2)

    def test_evaluation_preserves_training_and_rng(self):
        model = GPT(3, context_length=4, n_embd=8, n_head=2, n_layer=1)
        dataset = TextDataset([0, 1, 2] * 6, 4)
        state = torch.get_rng_state().clone()
        result = evaluate(model, dataset, batch_size=2, eval_batches=2)
        self.assertTrue(model.training)
        self.assertGreater(result["perplexity"], 0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
