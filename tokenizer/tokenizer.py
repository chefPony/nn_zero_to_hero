from typing import List, Tuple
from collections import Counter
import regex as re


class BasicTokenizer:

    def __init__(self):
        self.tokens = list()
        self._n_base_tokens = None
        self.vocab_size = 0

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        self.tokens = list(set(text.encode("utf-8")))
        self._n_base_tokens = len(self.tokens)
        self.vocab_size = vocab_size
        encoded_text = self.encode(text, self.tokens)
        while len(self.tokens) < vocab_size:
            bigram_counts = Counter([(c1, c2) for c1, c2 in zip(encoded_text, encoded_text[1:])])
            merge, count = bigram_counts.most_common(1)[0]
            self.tokens.append(merge)
            encoded_text = self.merge_tokens(encoded_text, merge, len(self.tokens) - 1)
            if verbose:
                token_1, token_2 = merge
                print(f"({self.tokens[token_1]} , {self.tokens[token_2]}) => {len(self.tokens) - 1}")

    def decode_token(self, idx):
        if idx < self._n_base_tokens:
            token = self.tokens[idx]
            return bytes([token])
        else:
            id1, id2 = self.tokens[idx]
            return b"".join([self.decode_token(id1), self.decode_token(id2)])

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

    def encode(self, text: str, tokens):
        encoded_text = text.encode("utf-8")
        c2idx = {c: idx for idx, c in enumerate(tokens)}
        encoded_text = [c2idx[c] for c in encoded_text]
        for i, (tok1, tok2) in enumerate(tokens[self._n_base_tokens:]):
            new_token = self._n_base_tokens + i
            encoded_text = self.merge_tokens(encoded_text, (tok1, tok2), new_token)
        return encoded_text

    def decode(self, encoded_text: List[int]):
        decode_dict = {i: self.decode_token(i) for i in range(len(self.tokens))}
        decoded_text = b''.join([decode_dict[t] for t in encoded_text])
        return decoded_text


class RegexTokenizer(BasicTokenizer):

    def __init__(self, regex = None):
        super().__init__()
        GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.regex = regex if regex else GPT4_SPLIT_PATTERN
        self._compiled_pattern = re.compile(self.regex)

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        self.tokens = list(set(text.encode("utf-8")))
        self._n_base_tokens = len(self.tokens)
        self.vocab_size = vocab_size
        encoded_text_splits = [self.encode(text_split, self.tokens) for text_split in re.findall(self._compiled_pattern, text)]
        while len(self.tokens) < vocab_size:
            bigram_counts = Counter([(c1, c2) for split in encoded_text_splits for c1, c2 in zip(split, split[1:])])
            merge, count = bigram_counts.most_common(1)[0]
            self.tokens.append(merge)
            encoded_text_splits = [self.merge_tokens(split, merge, len(self.tokens) - 1) for split in encoded_text_splits]
            if verbose:
                token_1, token_2 = merge
                print(f"({self.tokens[token_1]} , {self.tokens[token_2]}) => {len(self.tokens) - 1}")