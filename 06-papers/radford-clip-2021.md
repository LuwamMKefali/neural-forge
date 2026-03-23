# Learning Transferable Visual Models From Natural Language Supervision (CLIP)
**Radford et al. 2021** · [arxiv](https://arxiv.org/abs/2103.00020)  
**Read:** March 2026 · **Reproduction status:** Planned (Phase 2)

---

## Problem

Standard vision models are trained on fixed label sets — ImageNet has 1000 classes.
This limits what they can recognise and makes them brittle to distribution shift.

What if instead of discrete labels, you supervise vision with natural language?
The internet has hundreds of millions of image-text pairs. Can you use that?

## Key Idea

Train a vision encoder and a text encoder jointly so that matching image-text pairs
have similar embeddings, and non-matching pairs don't.

At inference — zero-shot classification by comparing the image embedding to text
embeddings of candidate class descriptions:

```
"a photo of a dog"
"a photo of a cat"
"a photo of a car"
          ↑
    pick the closest one
```

No task-specific fine-tuning needed.

## How It Works

**Dataset:** 400 million image-text pairs scraped from the internet (WIT).

**Architecture:**
- Image encoder: ResNet or ViT
- Text encoder: Transformer
- Both map to a shared embedding space of dimension d

**Training objective — contrastive loss:**

For a batch of N image-text pairs, create an N×N similarity matrix:

```
similarity[i][j] = cos_sim(image_embed[i], text_embed[j])
```

The diagonal is positive pairs (matching). Everything else is negative.

Loss: maximise diagonal similarities, minimise off-diagonal. This is symmetric
InfoNCE / NT-Xent applied across modalities:

```
L = (1/2) · (cross_entropy_over_rows + cross_entropy_over_columns)
```

Scaled by a learnable temperature τ.

### Why Contrastive and Not Generative?

The paper tries a predictive approach (predict exact caption from image) first —
it's 3x less efficient than contrastive. Contrastive training doesn't require
generating text, just aligning embeddings. Much faster to train at scale.

## Zero-Shot Transfer

At inference, for ImageNet classification:

1. For each class, create a prompt: `"a photo of a {class}"`
2. Embed all prompts with the text encoder → N text embeddings
3. Embed the query image with the vision encoder
4. Find the closest text embedding → predicted class

This is zero-shot — no ImageNet training data, no fine-tuning.

**Result:** 76.2% top-1 accuracy on ImageNet zero-shot. Matches a supervised ResNet-50.

## Connection to My Work

This is the most directly relevant paper to my BioMistral + VoRA work.

VoRA extends a language model to process images by injecting vision features via
LoRA-style adapters. CLIP shows that the right way to align vision and language is
through contrastive training on paired data — not through direct generation.

The projection head in CLIP (mapping both modalities to a shared space) is
architecturally similar to what the VoRA adapters do — they're both alignment mechanisms.

Understanding CLIP properly means I can reason about *why* VoRA works and what its
limitations are — the alignment between visual and language features is only as good
as the pre-training signal.

Also relevant to my bias auditing work: CLIP is known to inherit biases from
its training data at scale. The same texture shortcuts and demographic biases
I found in DeepFace/EmotiEffLib almost certainly exist in CLIP too.

## Questions I Still Have

- Prompt engineering matters a lot for zero-shot performance ("a photo of a {}" 
  outperforms just "{}"). Why? What does the text encoder do differently with
  natural language context vs bare class names?
- CLIP uses a symmetric loss — equal weight on image→text and text→image directions.
  Is this always optimal? What if the modalities are very asymmetric?
- ALIGN (Jia et al. 2021) achieves similar results with noisier but larger data
  (1.8B pairs vs 400M). What's the data quality vs quantity tradeoff here?

## Next Steps

- [ ] Reproduce a small CLIP on a subset dataset in `05-research-projects/clip-reproduction/`
- [ ] Read ALIGN (Jia et al. 2021) — scaling vs quality
- [ ] Explore CLIP's biases — connect to my bias auditing work
