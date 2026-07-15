# Run Log

All runs trained exclusively on `data/train_corpus.txt`.

---

## Run 1: Baseline Reference
* **Hypothesis**: The baseline byte-level tokenizer is inefficient because it treats each Devanagari character as 3 distinct tokens, causing the context window to fill up rapidly and wasting the model's capacity on spelling syllables. Additionally, a constant learning rate without warmup or decay under-optimizes model parameter adjustments.
* **What Changed**: Tested the baseline model architecture (4 layers, 4 heads, 160 embedding dim, GELU activation, LayerNorm, absolute positional embeddings) using raw UTF-8 bytes (vocabulary size 256). Used the baseline trainer with constant learning rate (3e-4, no decay, no warmup, no gradient clipping).
* **Dev BPB before/after**:
  * Before: N/A
  * After: **3.8214**
* **Conclusion**: The baseline setup converges poorly and has extremely low vocabulary compression. Syllable-splitting in Devanagari and constant learning rate limit bits-per-byte optimization on this corpus.

---

## Run 2: Custom BPE Tokenizer + Architectural Overhaul + Overhauled Trainer (Final ckpt.pt)
* **Hypothesis**:
  1. A custom BPE tokenizer with vocabulary size 2,048 using a Devanagari cluster-aware regex splitter (`[\p{L}\p{M}]+`) will group consonants and vowel marks together, increasing token compression ratio.
  2. Weight tying between embeddings and lm_head will free up parameter overhead to invest in model capacity.
  3. RMSNorm, SwiGLU activation, and Rotary Position Embeddings (RoPE) will accelerate convergence speed and capacity on CPU.
  4. AdamW optimizer with a cosine decay schedule, linear warmup, gradient clipping, and gradient accumulation (effective batch size 256 tokens) will stabilize training and maximize parameter efficiency in a 2000-step regime.
* **What Changed**:
  - Implemented BPE tokenizer with 2048 vocab and byte-fallback.
  - Replaced standard LayerNorm with RMSNorm, GELU with SwiGLU, absolute positional embedding with RoPE.
  - Tied `tok_emb` and `lm_head` weights.
  - Adjusted config defaults to `n_layer = 4`, `n_head = 6`, `n_embd = 180` (1,928,340 parameters).
  - Replaced Adam with AdamW, added decoupled decay, a 100-step linear warmup cosine schedule, gradient clipping at 1.0, and gradient accumulation.
* **Dev BPB before/after**:
  * Before: **3.8214** (Baseline)
  * After: **2.2622**
* **Conclusion**: Upgrades yielded a massive reduction in bits-per-byte, improving the score to **2.2622**. Tokenizer compression increased to 3.34 bytes/token, and the custom training dynamics allowed stable convergence on CPU within 2,000 steps.
