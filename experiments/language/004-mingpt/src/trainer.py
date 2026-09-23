"""Training loop in the style of karpathy/minGPT.

Upstream picks CUDA, otherwise CPU. This loop picks CUDA, then Apple MPS, then
CPU. pin_memory is only turned on for CUDA. The smoke sets num_workers to 0.
"""

import time

import torch
from torch.utils.data import DataLoader, Sampler


def pick_device():
    """CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ReplacementSampler(Sampler):
    """Draw dataset indices with replacement, a chunk at a time.

    Upstream uses RandomSampler(replacement=True, num_samples=1e10). A chunked
    sampler keeps that behavior without building one giant index tensor.
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        n = len(self.data_source)
        while True:
            yield from torch.randint(n, (1024,)).tolist()

    def __len__(self):
        return 2**31


class Trainer:
    def __init__(self, model, dataset, config, device=None):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.device = device or pick_device()
        self.model.to(self.device)
        self.optimizer = model.configure_optimizers(
            config["learning_rate"],
            config["weight_decay"],
            config["betas"],
        )
        self.iter_num = 0
        self.loss = None
        self.callbacks = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def run(self):
        pin_memory = self.device == "cuda"
        loader = DataLoader(
            self.dataset,
            sampler=ReplacementSampler(self.dataset),
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=pin_memory,
        )
        self.model.train()
        data_iter = iter(loader)
        started = time.perf_counter()
        while self.iter_num < self.config["max_iters"]:
            batch = next(data_iter)
            x, y = [tensor.to(self.device) for tensor in batch]
            _logits, self.loss = self.model(x, y)
            self.model.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_norm_clip"])
            self.optimizer.step()
            self.iter_num += 1
            for callback in self.callbacks:
                callback(self)
        return time.perf_counter() - started
