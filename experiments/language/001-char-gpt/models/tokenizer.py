"""Character vocabulary for Tiny Shakespeare.

Each character is one token. The vocabulary is the sorted set of characters
that appear in the training file, so index 0 is the newline.
"""


class CharTokenizer:
    """Maps characters to integer ids and back.

    `stoi` is string-to-index. `itos` is index-to-string. Both are built once
    from the corpus and saved in the checkpoint so sampling uses the same ids
    the model was trained with.
    """

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        # Sorting makes the ids stable across machines and Python versions.
        return cls(sorted(set(text)))

    @classmethod
    def from_state(cls, state):
        return cls(state["chars"])

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        """Turn a string into a list of token ids."""
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        """Turn token ids back into a string."""
        return "".join(self.itos[i] for i in ids)

    def state_dict(self):
        return {"chars": self.chars}
