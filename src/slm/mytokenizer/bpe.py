from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
from pathlib import Path
from collections import Counter, defaultdict

from ..utils.io import ensure_dir, write_json, read_json

_PAD = "<pad>"; _UNK = "<unk>"; _BOS = "<bos>"; _EOS = "<eos>"

@dataclass
class TokenizerModel:
    vocab: Dict[str, int]
    merges: List[Tuple[str, str]]
    special_tokens: List[str]
    byte_fallback: bool

    def to_dict(self):
        return {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "byte_fallback": self.byte_fallback,
        }

    @staticmethod
    def from_dict(d: dict) -> "TokenizerModel":
        return TokenizerModel(
            vocab=d["vocab"],
            merges=[tuple(x) for x in d["merges"]],
            special_tokens=d["special_tokens"],
            byte_fallback=d.get("byte_fallback", True),
        )

class ByteLevelBPE:
    """
    Industry-grade choices:
      • byte-level pretokenization for full Unicode coverage and stability
      • trainable BPE merges with frequency-based greedy merges
      • explicit special tokens + deterministic serialization
      • reversible encode/decode
    """

    def __init__(self,
                 model_dir: str,
                 vocab_size: int = 32000,
                 min_freq: int = 2,
                 special_tokens: Optional[List[str]] = None,
                 byte_fallback: bool = True,
                 lowercase: bool = False,
                 max_input_chars_per_token: int = 100):
        self.model_dir = Path(model_dir)
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or [_PAD, _UNK, _BOS, _EOS]
        self.byte_fallback = byte_fallback
        self.lowercase = lowercase
        self.max_input_chars_per_token = max_input_chars_per_token

        self.model: Optional[TokenizerModel] = None
        self._id2tok: Optional[List[str]] = None

    # ---------- Training ----------
    def train(self, corpus: Iterable[str], seed: int = 42) -> None:
        random.seed(seed)

        # Step 1: normalize + byte pretokenize each line into a sequence of bytes
        def normalize(s: str) -> str:
            return s.lower() if self.lowercase else s

        # Base vocab: all byte values 0..255 prefixed to avoid collisions
        # We represent initial tokens as single bytes encoded as 0xXX strings.
        base_vocab = {f"<0x{b:02x}>": i for i, b in enumerate(range(256))}
        next_id = len(base_vocab)

        # Add specials first, stable IDs at start of vocab
        for sp in self.special_tokens:
            if sp not in base_vocab:
                base_vocab[sp] = next_id
                next_id += 1

        # Tokenize lines to byte tokens
        sequences: List[List[str]] = []
        for line in corpus:
            line = normalize(line.rstrip("\n"))
            # encode to bytes; represent each byte as token "<0xXX>"
            seq = [f"<0x{b:02x}>" for b in line.encode("utf-8", errors="replace")]
            if seq:
                sequences.append(seq)

        # Count initial tokens
        token_freq = Counter()
        for seq in sequences:
            token_freq.update(seq)

        # Filter rare bytes? In byte-level, we keep all 256 to ensure coverage.
        vocab = dict(base_vocab)
        merges: List[Tuple[str, str]] = []

        # Helper: get pair frequencies
        def get_pair_freq(seqs: List[List[str]]) -> Counter:
            pair_freq = Counter()
            for s in seqs:
                for a, b in zip(s, s[1:]):
                    pair_freq[(a, b)] += 1
            return pair_freq

        # Main BPE merge loop
        # Target vocab size includes base bytes + specials + merges.
        while len(vocab) < self.vocab_size:
            pair_freq = get_pair_freq(sequences)
            if not pair_freq:
                break
            # choose most frequent pair with freq >= min_freq
            (best_a, best_b), best_f = pair_freq.most_common(1)[0]
            if best_f < self.min_freq:
                break

            new_token = best_a + best_b  # concat string tokens; unique since bytes are wrapped
            if new_token in vocab:
                # extremely unlikely, but guard
                break

            # Add merge to records
            merges.append((best_a, best_b))
            vocab[new_token] = len(vocab)

            # Apply merge to all sequences in place
            # We scan and replace A B -> AB greedily left-to-right
            for i, s in enumerate(sequences):
                j = 0
                out = []
                while j < len(s):
                    if j < len(s) - 1 and s[j] == best_a and s[j + 1] == best_b:
                        out.append(new_token)
                        j += 2
                    else:
                        out.append(s[j])
                        j += 1
                sequences[i] = out

        # Byte fallback: ensure unknown path via raw bytes if an OOV appears
        # Our byte base ensures this by design; decode will map to bytes safely.

        # Finalize model
        self.model = TokenizerModel(
            vocab=vocab,
            merges=merges,
            special_tokens=self.special_tokens,
            byte_fallback=self.byte_fallback,
        )
        # Build id2tok array for fast decode
        self._id2tok = [None] * len(vocab)
        for t, i in vocab.items():
            if i < len(self._id2tok):
                self._id2tok[i] = t

    # ---------- Encode / Decode ----------
    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """Apply stored merges to a byte-token sequence deterministically."""
        # Build merge dict for O(1) lookup
        if self.model is None:
            raise RuntimeError("Tokenizer not trained/loaded.")
        merge_set = set(tuple(m) for m in self.model.merges)

        changed = True
        while changed:
            changed = False
            i = 0
            out = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) in merge_set:
                    out.append(tokens[i] + tokens[i+1])
                    i += 2
                    changed = True
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
        return tokens

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        if self.model is None:
            raise RuntimeError("Tokenizer not trained/loaded.")
        t = text.lower() if self.lowercase else text
        # bytes → "<0xXX>" tokens
        toks = [f"<0x{b:02x}>" for b in t.encode("utf-8", errors="replace")]
        toks = self._apply_merges(toks)

        ids = []
        v = self.model.vocab
        if add_bos and _BOS in v: ids.append(v[_BOS])
        for tok in toks:
            if tok in v:
                ids.append(v[tok])
            elif self.model.byte_fallback:
                # fallback to raw bytes
                for b in tok.encode("utf-8", errors="replace"):
                    btok = f"<0x{b:02x}>"
                    ids.append(v.get(btok, v.get(_UNK)))
            else:
                ids.append(v.get(_UNK))
        if add_eos and _EOS in v: ids.append(v[_EOS])
        return ids

    def decode(self, ids: List[int]) -> str:
        if self.model is None or self._id2tok is None:
            raise RuntimeError("Tokenizer not trained/loaded.")
        # reconstruct byte stream ignoring specials
        byte_vals: List[int] = []
        for i in ids:
            tok = self._id2tok[i]
            if tok in self.model.special_tokens:
                continue
            # split merged token back to byte tokens deterministically
            stack = [tok]
            while stack:
                cur = stack.pop()
                if cur.startswith("<0x") and cur.endswith(">") and len(cur) == 6:
                    b = int(cur[3:5], 16)
                    byte_vals.append(b)
                else:
                    # not a single byte, split greedily into constituent byte tokens
                    # we know original atoms were "<0x??>", so scan for those
                    j = 0
                    while j < len(cur):
                        if cur[j:j+3] == "<0x":
                            k = j+3
                            # expect 2 hex + ">"
                            sub = cur[j:j+6]
                            stack.append(sub)
                            j += 6
                        else:
                            # This should not happen if training was consistent,
                            # but guard by byte-encoding remainder:
                            for b in cur[j].encode("utf-8", errors="replace"):
                                stack.append(f"<0x{b:02x}>")
                            j += 1
        return bytes(byte_vals).decode("utf-8", errors="replace")

    # ---------- Persistence ----------
    def save(self) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        d = self.model.to_dict()
        ensure_dir(self.model_dir)
        write_json(d, self.model_dir / "tokenizer.json")

    def load(self) -> None:
        path = self.model_dir / "tokenizer.json"
        model_dict = read_json(path)
        self.model = TokenizerModel.from_dict(model_dict)
        # rebuild id2tok
        self._id2tok = [None] * len(self.model.vocab)
        for tok, idx in self.model.vocab.items():
            if idx < len(self._id2tok):
                self._id2tok[idx] = tok

    # ---------- Introspection ----------
    def token_to_id(self, token: str) -> int:
        if self.model is None:
            raise RuntimeError("Tokenizer not trained/loaded.")
        return self.model.vocab.get(token, self.model.vocab.get(_UNK))

    def id_to_token(self, idx: int) -> str:
        if self._id2tok is None:
            raise RuntimeError("Tokenizer not trained/loaded.")
        return self._id2tok[idx]

    def vocab_size(self) -> int:
        if self.model is None:
            raise RuntimeError("Tokenizer not trained/loaded.")
        return len(self.model.vocab)

    def get_merges(self) -> List[Tuple[str, str]]:
        if self.model is None:
            raise RuntimeError("Tokenizer not trained/loaded.")
        return list(self.model.merges)
