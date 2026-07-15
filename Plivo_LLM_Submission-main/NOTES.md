# Design Decisions

* The primary setup implements a custom 2,048 vocabulary BPE tokenizer, tied weights, RMSNorm layer normalization, a SwiGLU MLP structure, and Rotary Position Embeddings (RoPE).
* This layout operates with 4 layers, 6 attention heads, and a 180-embedding dimension, resulting in a total of 1,928,340 parameters.
* A cluster-aware regex pattern `[\p{L}\p{M}]+` directs the BPE tokenizer to keep Devanagari consonants joined with their corresponding vocalic marks, attaining a compression performance of 3.34 bytes per token on the mixed dataset.
* An absolute byte-fallback map converts unknown characters to base bytes U+0000–U+00FF, preventing system crashes during evaluation on unexpected text.
* Sharing parameters between the token embedding and the linear projection head eliminates duplicate parameters, letting the network dedicate more capacity to representation learning.
* Substituting RMSNorm for LayerNorm increases training speed and stability on CPU by omitting mean value computations.
* The feedforward sections utilize SwiGLU activation functions to introduce multiplicative gating, which substantially accelerates convergence.
* Query and key rotations inside the SelfAttention layer are handled by Rotary Position Embeddings (RoPE), providing relative positional clues and enabling generalization to longer sequence lengths.
* Decoupled weight decay applied via the AdamW optimizer (excluding bias terms and normalization layers) mitigates overfitting throughout the 2,000-step training window.
* The optimizer employs a cosine schedule with a 100-step linear warmup to guarantee gradual learning step updates and smooth loss reduction.
* Gradient accumulation processes micro-batches of size 1 to achieve an effective batch size of 256 tokens without exceeding CPU memory limits.
