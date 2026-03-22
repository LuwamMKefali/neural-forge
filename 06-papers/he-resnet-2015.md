# Deep Residual Learning for Image Recognition
**He et al. 2015** · [arxiv](https://arxiv.org/abs/1512.03385)  
**Read:** March 2026 · **Reproduction status:** Not started

---

## Problem

As networks get deeper, they get harder to train — not because of overfitting, but because of
something more fundamental. A deeper network should in theory perform at least as well as a
shallower one (the extra layers could just learn identity mappings). In practice they don't.
Training error actually goes *up* with depth. This is the degradation problem.

It's not vanishing gradients alone — this was observed even with BatchNorm in place.

## Key Idea

Instead of learning a direct mapping `H(x)`, let the layers learn the **residual**:

```
F(x) = H(x) - x
```

Then the output is:

```
H(x) = F(x) + x
```

The `+ x` is the skip connection — it shortcuts the input directly to the output.

If the optimal mapping is close to identity, it's much easier to push `F(x)` toward 0
than to learn an identity mapping from scratch through a stack of nonlinear layers.

## Why It Works — The Math

### Gradient Flow

During backprop, the gradient through a residual block is:

```
dL/dx = dL/dH · (dF/dx + 1)
```

The `+1` term means the gradient always has a direct path back through the skip connection,
even if `dF/dx` is small. This is what prevents vanishing gradients in very deep networks.

### The Skip Connection

Two cases in the paper:

**Case 1 — same dimensions:** just add x directly
```
output = F(x, {Wi}) + x
```

**Case 2 — different dimensions** (when downsampling with stride):
```
output = F(x, {Wi}) + Ws · x
```
where `Ws` is a 1×1 convolution to match dimensions.

## Architecture

A residual block (two 3×3 conv layers):
```
x → [Conv 3x3 → BN → ReLU → Conv 3x3 → BN] → + x → ReLU
```

For very deep networks (50+ layers), they use a bottleneck block instead:
```
x → [Conv 1x1 → BN → ReLU → Conv 3x3 → BN → ReLU → Conv 1x1 → BN] → + x → ReLU
```
The 1x1 convs reduce and restore dimensions — much cheaper to compute.

## Results

- ResNet-152 achieved 3.57% top-5 error on ImageNet — won ILSVRC 2015
- Training a 1,202-layer network was possible (though it overfit)
- Deeper = better, as long as you have residuals

## Connection to My Work

I ran a controlled ablation on skip connections in `PetResNet_v1` — removing them dropped
accuracy from ~60.95% to baseline. At the time I reported the number without fully
understanding *why*. Now I understand it: without the skip connection, the gradient
path through 3+ layers degrades, making it much harder for deep blocks to learn.

The bottleneck design is also directly relevant to LoRA — both are based on the idea
that useful transformations can live in a lower-dimensional space than the full layer width.

## Questions I Still Have

- The paper shows that even very deep plain networks (no residuals) have higher training
  error than shallower ones. Why exactly? BatchNorm should handle vanishing gradients.
  Is it an optimisation landscape issue rather than a gradient issue?
- Pre-activation ResNets (He et al. 2016 follow-up) put BN and ReLU *before* the conv
  rather than after. That reportedly works better — worth reading why.
- DenseNet (Huang et al. 2017) extends this idea with connections to *all* previous layers,
  not just the previous block. How does that compare?

## Next Steps

- [ ] Implement a ResNet block from scratch in `03-computer-vision/`
- [ ] Re-run my pet classification ablation and write it up properly with this understanding
- [ ] Read He et al. 2016 — Identity Mappings in Deep Residual Networks (Pre-activation)
