# LoRA: Low-Rank Adaptation of Large Language Models
**Hu et al. 2021** · [arxiv](https://arxiv.org/abs/2106.09685)  
**Read:** March 2026 · **Reproduction status:** In progress

---

## Problem
Full fine-tuning of large pre-trained models is expensive — you update every parameter,
which means storing a full copy of the model per task. For a model like GPT-3 with 175B
parameters, that's completely impractical.

## Key Idea
Instead of updating the full weight matrix **W**, freeze it and inject a trainable
low-rank decomposition alongside it:

```
W' = W + ΔW = W + BA
```

where **B** ∈ ℝ^(d×r) and **A** ∈ ℝ^(r×k), with rank r << min(d, k).

Only **A** and **B** are trained. **W** never changes.

## Why It Works — The Math

The hypothesis is that the weight updates during fine-tuning have a low intrinsic rank.
In other words, ΔW doesn't need to be a full-rank matrix to adapt the model to a new task.

**Initialisation:**
- A is initialised with random Gaussian
- B is initialised to zero → so ΔW = BA = 0 at the start of training
- This means training begins from the original pre-trained behaviour. Clean.

**Scaling:**
The update is scaled by α/r where α is a hyperparameter:

```
output = Wx + (α/r) · BAx
```

This keeps the magnitude of the update stable regardless of what rank r you choose.

**Parameter count:**
A full weight matrix d×k has d·k parameters.
LoRA replaces the update with r·(d+k) parameters.
For r=8, d=768, k=768: 8·1536 = 12,288 vs 589,824. ~48x fewer parameters.

## Results
- Matches or outperforms full fine-tuning on GPT-3 benchmarks
- Applied to attention matrices: Q, K, V, and output projection
- r=4 or r=8 is enough for most tasks — higher rank rarely helps

## Connection to My Work
I used LoRA via PEFT for BioMistral-7B + VoRA. At the time I treated it as a
configuration choice. Reading the paper properly, I now understand that the VoRA
adapters work on exactly the same principle — injecting low-rank updates into
specific weight matrices while keeping the base model frozen.

The reason it works for vision-language adaptation isn't magic — it's that the
visual feature alignment task has a low intrinsic dimensionality relative to the
full weight space.

## Questions I Still Have
- What's the theoretical justification for the low intrinsic rank hypothesis?
  The paper cites Aghajanyan et al. 2020 — read that next.
- How does LoRA interact with quantisation (QLoRA)? Follow-up: Dettmers et al. 2023.
- Why do Q and V matrices benefit more than K? The paper observes this empirically
  but doesn't fully explain it.

## Next Steps
- [ ] Implement LoRA from scratch in `02-nlp-depth/lora-deep-dive/`
- [ ] Read Aghajanyan et al. 2020 — intrinsic dimensionality
- [ ] Read QLoRA paper (Dettmers et al. 2023)
