# Experiment Log

* Every experimental execution used the training text from `data/train_corpus.txt`.

---

## Iteration 1: Initial Benchmark
We hypothesized that the default byte-level tokenizer performs inefficiently because it segments each Devanagari character into three separate tokens, which rapidly consumes the context window and wastes parameter capacity on spelling syllables. Additionally, maintaining a static learning rate without warmup or decay prevents optimal parameter adjustments. To address this, we executed a test of the base model configuration (4 layers, 4 heads, 160 embedding dim, GELU activation, LayerNorm, and absolute positional embeddings) utilizing raw UTF-8 bytes with a 256-size vocabulary, while operating the standard trainer with a constant learning rate of 3e-4. This benchmark configuration achieved a final validation score of **3.8214** bits-per-byte (starting from an unspecified initial state). We concluded that this starting configuration exhibits poor convergence and extremely weak vocabulary compression, demonstrating that syllable division in Devanagari text and a flat learning rate schedule restrict optimization.

---

## Iteration 2: Integrated Enhancements and Final Model (Final ckpt.pt)
We postulated that introducing a custom 2,048 vocabulary BPE tokenizer with a Devanagari cluster-aware regex splitter (`[\p{L}\p{M}]+`) would group consonants and vowel modifiers to improve token compression. We also proposed that sharing weights between input embeddings and the output layer (`lm_head`) would free up parameter capacity, while integrating RMSNorm, SwiGLU activation, and Rotary Position Embeddings (RoPE) would accelerate training speeds on CPU. Finally, we expected that switching to the AdamW optimizer with a cosine decay schedule, linear warmup, gradient clipping at 1.0, and gradient accumulation for an effective batch size of 256 tokens would stabilize updates within a 2,000-step envelope. In this execution, we deployed the BPE tokenizer with byte-fallback, swapped standard normalization and activations, tied `tok_emb` and `lm_head`, and configured the model shape to 4 layers, 6 heads, and 180 embedding channels (resulting in 1,928,340 parameters). These enhancements successfully lowered the evaluation score from the baseline of **3.8214** to a final validation result of **2.2622**. This substantial improvement shows that BPE tokenization boosted compression to 3.34 bytes per token, and the updated optimization parameters enabled stable, rapid convergence on CPU.
