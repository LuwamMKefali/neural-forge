# Attention Is All You Need
**Vaswani et al. 2017** · [arxiv](https://arxiv.org/abs/1706.03762)  
**Read:** March 2026 · **Reproduction status:** In progress

---

## Problem
Before this paper, sequence modelling relied on RNNs and LSTMs — which process tokens
sequentially. That means two things: slow to train (can't parallelise), and bad at
capturing long-range dependencies (gradients vanish over long sequences).

Convolutions helped with parallelism but still struggled with long-range context.
The question was: can you build a sequence model with no recurrence at all?

## Key Idea
Replace recurrence entirely with **self-attention** — let every token attend directly
to every other token in the sequence, regardless of distance.

The result is the **Transformer**: an encoder-decoder architecture built purely from
attention and feed-forward layers.

## The Core Mechanism — Scaled Dot-Product Attention

Given queries **Q**, keys **K**, and values **V**:

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
```

**Breaking it down:**
- `QK^T` — dot product between every query and every key. Measures compatibility.
- `/ sqrt(d_k)` — scale down to prevent softmax from saturating in high dimensions.
  Without this, large dot products push softmax into regions with near-zero gradients.
- `softmax(...)` — converts scores to a probability distribution over positions.
- `· V` — weighted sum of values. High-attention positions contribute more.

**Why sqrt(d_k) specifically?**
If Q and K have unit variance, QK^T has variance d_k. Dividing by sqrt(d_k)
brings it back to unit variance. Clean.

## Multi-Head Attention

Instead of one attention function, run h attention functions in parallel on
projected subspaces, then concatenate:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
where head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)
```

Each head learns to attend to different types of relationships simultaneously —
one might track syntactic structure, another coreference, another positional patterns.

## Positional Encoding
Attention has no notion of order — it treats the input as a set, not a sequence.
To inject position information, they add sinusoidal encodings to the input embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

The choice of sinusoids lets the model generalise to sequence lengths not seen during training.

## Architecture
```
Input → Embedding + Positional Encoding
      → N × [Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm]  (Encoder)
      → N × [Masked MHA → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm] (Decoder)
      → Linear → Softmax → Output
```

- **Add & Norm** = residual connection + layer normalisation
- **FFN** = two linear layers with ReLU: FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
- N = 6 layers in the original paper

## Results
- Achieved state-of-the-art on WMT 2014 EN→DE and EN→FR translation
- Trained in a fraction of the time of previous SOTA models
- 28.4 BLEU on EN→DE — best result at the time

## Connection to My Work
Every model I've worked with — BERT, BioMistral, the VoRA adapters — is built on
exactly this architecture. I've used multi-head attention hundreds of times.

What I hadn't done before reading this properly: understood *why* the scaling works,
*why* sinusoidal encodings, and *why* multi-head rather than one big attention.
The answers are all in the math above.

Next step is implementing this from scratch in `02-nlp-depth/attention-from-scratch/`.

## Questions I Still Have
- The paper uses learned positional embeddings in some variants — when is sinusoidal
  better and when is learned better? RoPE (used in LLaMA) is a more modern answer.
- Why Layer Norm after the residual and not before? Later work (Pre-LN transformers)
  flips this and shows it stabilises training — worth reading.
- What does each attention head actually learn? Visualisation work by Clark et al. 2019
  is the follow-up here.

## Next Steps
- [ ] Implement scaled dot-product attention from scratch — `02-nlp-depth/attention-from-scratch/`
- [ ] Implement full multi-head attention block
- [ ] Read Clark et al. 2019 — what do attention heads learn?
- [ ] Read Pre-LN transformer paper — why Pre-LN stabilises training
