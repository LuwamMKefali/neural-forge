# A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
**Chen et al. 2020** · [arxiv](https://arxiv.org/abs/2002.05709)  
**Read:** March 2026 · **Reproduction status:** Planned (Phase 2)

---

## Problem

Supervised learning needs labelled data. Labels are expensive. Can we learn good visual
representations from *unlabelled* images alone?

This is self-supervised learning — and SimCLR is one of the cleanest formulations of it.

## Key Idea

If you take the same image and apply two different random augmentations, the resulting
views should be close together in embedding space. All other images in the batch should
be pushed away.

That's it. No labels. Just: *same image → similar representation, different image → different representation.*

## How It Works

**Step 1 — Augmentation**  
For each image x, sample two augmented views: x_i and x_j.
Augmentations used: random cropping, colour distortion, Gaussian blur.

**Step 2 — Encode**  
Pass both through the same encoder f (a ResNet):
```
h_i = f(x_i),  h_j = f(x_j)
```

**Step 3 — Project**  
Pass through a small MLP projection head g:
```
z_i = g(h_i),  z_j = g(h_j)
```
The loss is computed in z-space. The representation h is what you actually use downstream.

**Step 4 — Contrastive Loss (NT-Xent)**

For a batch of N images → 2N augmented views.
For each positive pair (z_i, z_j), the loss pushes them together while pushing the
2(N-1) negatives away:

```
L(i,j) = -log [ exp(sim(z_i, z_j) / τ) / sum_{k≠i} exp(sim(z_i, z_k) / τ) ]
```

where:
- `sim(u, v) = u·v / (||u|| ||v||)` — cosine similarity
- `τ` — temperature hyperparameter (controls how sharp the distribution is)

The total loss is averaged over all positive pairs in the batch.

## Why the Projection Head?

This surprised me. You'd think using h directly for the loss would be best — it's the
representation you want to improve. But empirically, adding the projection head g and
computing the loss on z consistently outperforms not having it.

The paper's explanation: the projection head allows the network to discard information
in the contrastive loss that's actually useful for downstream tasks. By separating the
representation h from the loss space z, the encoder isn't forced to compress useful
information away.

## What Matters Most

The paper runs extensive ablations:

| Factor | Finding |
|--------|---------|
| Augmentation | Colour distortion + random crop is critical. Without it, the model shortcuts on colour histograms. |
| Batch size | Larger = better. More negatives = harder task = better representation. |
| Temperature τ | Needs tuning. Too high = loss too uniform. Too low = collapses on hard negatives. |
| Projection head depth | 2-layer MLP is better than 1 or 0 layers. |

## Results

- Achieves 76.5% top-1 on ImageNet with a linear classifier on top — no labels during training
- With fine-tuning, matches or beats supervised ResNet-50
- Large batch (4096) + strong augmentation + projection head = the recipe

## Connection to My Work

I haven't done contrastive learning directly yet, but it's relevant in two ways:

1. My bias auditing work uses Grad-CAM to understand *what* the model attends to. SimCLR
   representations are known to be less texture-biased than supervised models — which means
   they might generalise better across neurodivergent populations. Worth exploring.

2. The projection head trick (separate representation from loss space) is architecturally
   similar to what VoRA does — keeping the base model's representation intact while
   the adapter head handles task-specific adaptation.

## Questions I Still Have

- Why does cosine similarity work better than dot product here? Both measure alignment,
  but cosine normalises for magnitude. Is magnitude a distractor in this case?
- MoCo (He et al. 2020) achieves similar results with a much smaller batch using a
  momentum encoder and a memory queue. How do the two compare on small compute budgets?
- BYOL (Grill et al. 2020) gets rid of negatives entirely and still works. How? That
  seems like it should collapse.

## Next Steps

- [ ] Reproduce SimCLR on CIFAR-10 in `03-computer-vision/simclr-reproduction/`
- [ ] Read MoCo — momentum encoder as an alternative to large batches
- [ ] Read BYOL — understanding why it doesn't collapse without negatives
