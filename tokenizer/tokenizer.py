from typing import List, Tuple
from collections import Counter


class BasicTokenizer:

    def __init__(self):
        self.tokens = list()
        self._n_base_tokens = None
        self.vocab_size = 0

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        self.vocab_size = vocab_size
        encoded_text = self.encode(text)
        while len(self.tokens) < vocab_size:
            bigram_counts = Counter([(c1, c2) for c1, c2 in zip(encoded_text, encoded_text[1:])])
            merge, count = bigram_counts.most_common(1)[0]
            self.tokens.append(merge)
            encoded_text = self.merge_tokens(encoded_text, merge, len(self.tokens) - 1)
            if verbose:
                token_1, token_2 = merge
                print(f"({self.tokens[token_1]} , {self.tokens[token_2]}) => {len(self.tokens) - 1}")

    def show_token(self, idx):
        if idx < self._n_base_tokens:
            token = self.tokens[idx]
            return bytes([token])
        else:
            id1, id2 = self.tokens[idx]
            return b"".join([self.show_token(id1), self.show_token(id2)])
    @staticmethod
    def merge_tokens(encoded_text: List[int], merge: Tuple[int, int], new_idx: int):
        # aba ab
        k = 1
        new_encoded = list()
        loop_text = encoded_text + [None]
        while k < len(loop_text):
            idx0, idx1 = loop_text[k-1], loop_text[k]
            if merge == (idx0, idx1):
                new_encoded.append(new_idx)
                k += 2
            else:
                new_encoded.append(idx0)
                k += 1
        return new_encoded

    def decode_token(self, encoded_text: List[int], idx: int):
        to_split = self.tokens[idx]
        if isinstance(to_split, tuple):
            after_split = list()
            for token in encoded_text:
                if token == idx:
                    after_split += list(to_split)
                else:
                    after_split.append(token)
            return after_split
        else:
            return [self.tokens[idx] if token == idx else token for token in encoded_text]

    def encode(self, text: str):
        encoded_text = text.encode("utf-8")
        self.tokens = list(set(encoded_text))
        self._n_base_tokens = len(self.tokens)
        c2idx = {c: idx for idx, c in enumerate(self.tokens)}
        encoded_text = [c2idx[c] for c in encoded_text]
        for idx, replace in enumerate(self.tokens[self._n_base_tokens + 1:]):
            encoded_text = self.merge_tokens(encoded_text, replace, idx)
        return encoded_text

    def decode(self, encoded_text: List[int]):
        decoded_text = encoded_text
        for i in range(len(self.tokens)):
            idx = len(self.tokens) - i - 1
            decoded_text = self.decode_token(decoded_text, idx)
        return bytes(decoded_text)

