"""Byte-Pair Encoding (BPE) tokenizer trained on the corpus data.
Handles Devanagari script efficiently by preserving character clusters,
maintaining a strict byte fallback for arbitrary UTF-8 encoding.
"""
import os
import json
from collections import defaultdict, Counter

try:
    import regex
except ImportError:
    regex = None


class ByteTokenizer:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"type": "byte"}, f)


class BPETokenizer:
    def __init__(self, merges_path=None):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.cache = {}
        if merges_path and os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for i, pair in enumerate(data.get("merges", [])):
                    p = tuple(pair)
                    self.merges[p] = 256 + i
                    self.vocab[256 + i] = self.vocab[p[0]] + self.vocab[p[1]]
        self.vocab_size = 256 + len(self.merges)

    def encode(self, text):
        if not text:
            return []
        if regex is None:
            raise ImportError("The 'regex' library is required for BPETokenizer.")
        pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+""")
        words = pat.findall(text)
        
        ids = []
        for word in words:
            if word in self.cache:
                ids.extend(self.cache[word])
            else:
                word_bytes = list(word.encode("utf-8"))
                word_ids = self._encode_word(word_bytes)
                self.cache[word] = word_ids
                ids.extend(word_ids)
        return ids

    def _encode_word(self, word_bytes):
        if len(word_bytes) < 2:
            return word_bytes
        ids = list(word_bytes)
        while len(ids) >= 2:
            best_pair = None
            best_rank = float('inf')
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                rank = self.merges.get(pair, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_pair = pair
            if best_pair is None or best_rank == float('inf'):
                break
            # Merge
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                    new_ids.append(self.merges[best_pair])
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return ids

    def decode(self, ids):
        byte_segments = []
        for idx in ids:
            if idx in self.vocab:
                byte_segments.append(self.vocab[idx])
            else:
                if 0 <= idx < 256:
                    byte_segments.append(bytes([idx]))
        b = b"".join(byte_segments)
        return b.decode("utf-8", errors="replace")

    def save(self, path):
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        merges_list = [list(pair) for pair, _ in sorted_merges]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": merges_list}, f)


def load(path=None):
    """Return the tokenizer used by evaluate.py. Replace as needed."""
    if path is None:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dir_path, "bpe_merges.json")
    
    if os.path.exists(path):
        return BPETokenizer(path)
    else:
        print(f"Warning: BPE merges file not found at {path}. Falling back to ByteTokenizer.")
        return ByteTokenizer()


def train_bpe(corpus_path, out_merges_path, vocab_size=2048):
    import time
    if regex is None:
        raise ImportError("The 'regex' library is required for BPE training.")
        
    t0 = time.time()
    print(f"Loading training corpus from: {corpus_path}")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Corpus loaded: {len(text)} characters, {len(text.encode('utf-8'))} bytes.")
    
    t1 = time.time()
    pat = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}\p{M}]+| ?\p{N}+| ?[^\s\p{L}\p{M}\p{N}]+|\s+(?!\S)|\s+""")
    words = pat.findall(text)
    print(f"Regex split {len(words)} words ({len(set(words))} unique) in {time.time() - t1:.2f}s")
    
    t1 = time.time()
    word_freqs = Counter(words)
    vocab_words = {}
    word_counts = {}
    for word, count in word_freqs.items():
        w_bytes = tuple(word.encode("utf-8"))
        vocab_words[w_bytes] = list(w_bytes)
        word_counts[w_bytes] = count
        
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    for w_bytes, count in word_counts.items():
        tokens = vocab_words[w_bytes]
        for pair in zip(tokens, tokens[1:]):
            pair_counts[pair] += count
            pair_to_words[pair].add(w_bytes)
    print(f"Initialized pair maps in {time.time() - t1:.2f}s")
    
    t1 = time.time()
    num_merges = vocab_size - 256
    merges = []
    
    for step in range(num_merges):
        if not pair_counts:
            print("No active pairs left!")
            break
            
        best_pair = max(pair_counts, key=pair_counts.get)
        new_id = 256 + step
        merges.append(best_pair)
        
        words_to_update = list(pair_to_words[best_pair])
        
        if best_pair in pair_counts:
            del pair_counts[best_pair]
        if best_pair in pair_to_words:
            del pair_to_words[best_pair]
            
        for w in words_to_update:
            count = word_counts[w]
            tokens = vocab_words[w]
            
            for pair in zip(tokens, tokens[1:]):
                if pair == best_pair:
                    continue
                pair_counts[pair] -= count
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(w)
                
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            vocab_words[w] = new_tokens
            
            for pair in zip(new_tokens, new_tokens[1:]):
                pair_counts[pair] += count
                pair_to_words[pair].add(w)
                
        if (step + 1) % 200 == 0 or step == 0 or step == num_merges - 1:
            print(f"Step {step+1}/{num_merges}: merged {best_pair} -> {new_id}")
            
    print(f"BPE training completed in {time.time() - t1:.2f}s (Total: {time.time() - t0:.2f}s)")
    
    merges_list = [list(pair) for pair in merges]
    with open(out_merges_path, "w", encoding="utf-8") as f:
        json.dump({"merges": merges_list}, f)
    print(f"Saved merges table to {out_merges_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=None, help="Path to training corpus")
    parser.add_argument("--out", default=None, help="Path to output merges file")
    args = parser.parse_args()
    
    dir_path = os.path.dirname(os.path.abspath(__file__))
    if args.corpus is None:
        path1 = os.path.join(dir_path, "data", "train_corpus.txt")
        path2 = os.path.join(dir_path, "..", "data", "train_corpus.txt")
        args.corpus = path1 if os.path.exists(path1) else path2
    if args.out is None:
        args.out = os.path.join(dir_path, "bpe_merges.json")
        
    train_bpe(args.corpus, args.out)
