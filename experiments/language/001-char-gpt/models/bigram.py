"""Bigram language model: a baseline that sees only the current character.

The embedding table has shape (vocab_size, vocab_size). Row i is the vector of
logits for "what character comes after character i". There is no attention and
no use of earlier context, which is why this model is the comparison point for
the GPT in gpt.py.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        # One row of logits per character. 65 characters -> 65 * 65 = 4,225 parameters.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """idx and targets are integer tensors of shape (batch, time).

        Loss is the average cross-entropy over every position in the block.
        The prediction at each position still depends only on the token sitting
        there, not on the tokens before it.
        """
        logits = self.token_embedding_table(idx)  # (batch, time, vocab)
        loss = None
        if targets is not None:
            batch, time, channels = logits.shape
            loss = F.cross_entropy(logits.view(batch * time, channels), targets.view(batch * time))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Append max_new_tokens characters, one at a time.

        Only the last column of logits is used. Earlier tokens stay in `idx`
        so the returned tensor is the prompt plus the continuation.
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # (batch, vocab)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
