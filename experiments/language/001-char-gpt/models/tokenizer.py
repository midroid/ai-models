"""Character-level vocabulary built from a training corpus."""


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @classmethod
    def from_state(cls, state):
        return cls(state["chars"])

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def state_dict(self):
        return {"chars": self.chars}
