# Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
**Selvaraju et al. 2017** · [arxiv](https://arxiv.org/abs/1610.02391)  
**Read:** March 2026 · **Reproduction status:** Done (used in IEEE paper)

---

## Problem

Deep CNNs are black boxes. They give you a prediction but no explanation of *why*.
For high-stakes applications — medical imaging, autonomous driving, bias auditing —
that's not good enough.

The question: can you produce a heatmap showing which parts of the image were most
important for a given prediction, without modifying the model?

## Key Idea

Use the gradients of the class score with respect to the final convolutional feature
maps to understand which spatial regions the network focused on.

Regions with large positive gradients contributed most to the prediction.

## The Math

For a target class c, the class activation map is:

**Step 1 — Compute importance weights:**

```
α_k^c = (1/Z) · sum_{i,j} (∂y^c / ∂A^k_{ij})
```

- `y^c` — score for class c (before softmax)
- `A^k` — feature map k from the last conv layer
- `Z` — number of pixels (normalisation)
- This is global average pooling of the gradients over the spatial dimensions

**Step 2 — Weighted combination:**

```
L^c_Grad-CAM = ReLU( sum_k α_k^c · A^k )
```

The ReLU keeps only features that have a positive influence on the class score —
negative activations would correspond to evidence for other classes.

**Step 3 — Upsample** to input resolution and overlay on the image.

## Why the Last Conv Layer?

The last convolutional layer has the best tradeoff between:
- **Spatial resolution** — still has some spatial information
- **Semantic content** — deep enough to encode high-level features

Earlier layers have higher resolution but encode low-level features (edges, textures).
Fully connected layers have lost spatial information entirely.

## Guided Grad-CAM

Grad-CAM highlights broad regions but isn't pixel-sharp. Combining it with
Guided Backpropagation (which gives pixel-level gradients) produces Guided Grad-CAM —
high-resolution and class-discriminative.

```
Guided Grad-CAM = Guided Backprop ⊙ Grad-CAM (upsampled)
```

## Connection to My Work

I used Grad-CAM directly in my bias auditing paper. The finding: DeepFace and EmotiEffLib
were attending to texture regions around the mouth and eyes rather than the holistic
facial expression — and this shortcut generalised poorly to autism subsets where
facial expressions differ in texture patterns.

Now reading the paper properly, I understand the mechanism behind what I observed.
The gradient signal I was visualising was `∂(emotion score) / ∂(feature maps)` —
and the high activations in texture regions meant those gradients were large there.
The model had learned to rely on a texture-based shortcut.

LIME gave complementary information — superpixel-level attribution rather than
feature-map-level. Both pointed to the same failure mode, which is what made
the finding robust enough to publish.

## Limitations I've Noticed

- Grad-CAM highlights *where* the model looks, not *why* that region matters semantically
- It's sensitive to the choice of target layer — different layers give different maps
- For models without conv layers (ViTs, transformers), Grad-CAM doesn't apply directly.
  Attention rollout or transformer-specific attribution methods are needed instead.
- Class leakage — the gradients for class A can overlap spatially with features
  relevant to class B, making the maps less crisp for fine-grained tasks

## Questions I Still Have

- Attention rollout (Abnar & Zuidema 2020) gives attribution maps for transformers
  by propagating attention weights through layers. How does it compare to Grad-CAM
  for ViT models?
- Score-CAM (Wang et al. 2020) removes the gradient entirely and uses forward passes
  with masked feature maps instead. Is it actually better, or just slower?
- In my bias auditing work I used Grad-CAM qualitatively. How would I run a
  quantitative evaluation of the attribution quality?

## Next Steps

- [ ] Apply Grad-CAM to ViT and compare to attention rollout in `03-computer-vision/`
- [ ] Write a proper methods section on Grad-CAM + LIME for the bias auditing extension
- [ ] Read Score-CAM (Wang et al. 2020)
