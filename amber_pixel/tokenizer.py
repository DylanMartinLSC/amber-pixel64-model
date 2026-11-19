"""Word-level tokenizer built from the training captions alone (no pretrained vocabulary)."""
import json
import re


def words(text):
    return re.findall(r"[\w]+(?:'[\w]+)?|[^\w\s]", text.lower(), flags=re.UNICODE)


class Tokenizer:
    PAD, BOS, EOS, UNK = 0, 1, 2, 3

    def __init__(self, tokens, max_length=48):
        self.tokens, self.max_length = tokens, max_length
        self.vocabulary = {w: i for i, w in enumerate(tokens)}

    @classmethod
    def load(cls, path):
        data = json.loads(open(path, encoding='utf-8').read())
        return cls(data['tokens'], data['max_length'])

    def encode(self, text):
        ids = [self.BOS] + [self.vocabulary.get(w, self.UNK) for w in words(text)[:self.max_length - 2]] + [self.EOS]
        return ids + [self.PAD] * (self.max_length - len(ids))

    def unknown_words(self, text):
        return [w for w in words(text) if w not in self.vocabulary]

    def null(self):
        """The empty caption used for classifier-free guidance."""
        return [self.BOS, self.EOS] + [self.PAD] * (self.max_length - 2)
