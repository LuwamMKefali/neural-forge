# Denoising Diffusion Probabilistic Models (DDPM)
**Ho et al. 2020** · [arxiv](https://arxiv.org/abs/2006.11239)  
**Read:** March 2026 · **Reproduction status:** Not started

---

## Problem

Generative modelling — learning a distribution p(x) from data so you can sample new x.

GANs do this but are notoriously unstable to train. VAEs are stable but produce blurry
samples because of the ELBO approximation. Flow models are exact but architecturally
constrained.

Diffusion models offer a different approach: learn to reverse a gradual noising process.

## Key Idea

**Forward process:** gradually destroy an image by adding Gaussian noise over T steps
until it's pure noise.

**Reverse process:** learn a neural network that denoises step by step — going from
pure noise back to a clean image.

If you can learn the reverse, you can generate by starting from noise and denoising.

## The Math

### Forward Process (fixed, not learned)

At each step t, add a small amount of Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; sqrt(1 - β_t) · x_{t-1}, β_t · I)
```

where β_t is a variance schedule (small constants, linearly increasing from β_1 to β_T).

**Nice property** — you can sample x_t at any step t directly from x_0:

```
q(x_t | x_0) = N(x_t; sqrt(ᾱ_t) · x_0, (1 - ᾱ_t) · I)
```

where ᾱ_t = product of (1 - β_s) for s=1 to t.

In practice:
```
x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε,   ε ~ N(0, I)
```

So you can get a noisy version at any timestep in one shot — no need to chain T steps.

### Reverse Process (learned)

The reverse `p_θ(x_{t-1} | x_t)` is what the neural network learns.

It's parameterised as Gaussian:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

Rather than predicting the mean directly, the paper finds it works better to predict
the noise ε that was added:

```
ε_θ(x_t, t) ≈ ε
```

### Training Objective

Simplified loss — predict the noise:

```
L = E_{t, x_0, ε} [ || ε - ε_θ(sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε, t) ||² ]
```

At each training step:
1. Sample a clean image x_0
2. Sample a random timestep t
3. Sample noise ε
4. Compute noisy x_t
5. Predict the noise with the network
6. Minimise MSE between predicted and actual noise

### Sampling (inference)

Start from x_T ~ N(0, I), then iteratively denoise:

```
x_{t-1} = (1/sqrt(α_t)) · (x_t - (β_t / sqrt(1 - ᾱ_t)) · ε_θ(x_t, t)) + σ_t · z
```

where z ~ N(0, I) and σ_t is a noise scale. T=1000 steps in the original paper.

## Architecture

The noise prediction network is a **U-Net** with:
- Residual blocks
- Self-attention at lower resolutions
- Timestep t embedded via sinusoidal encoding (same as positional encoding) and injected into each residual block

## Connection to My Work

I haven't worked with diffusion models directly, but understanding the score-matching
interpretation connects it to representation learning — and diffusion-based approaches
are increasingly being applied to multimodal generation, which is directly relevant
to my VLM work.

The sinusoidal timestep embedding is identical to the positional encoding I just
implemented — same principle, different domain. That connection is clean.

## Questions I Still Have

- The paper uses T=1000 denoising steps which is extremely slow at inference.
  DDIM (Song et al. 2020) achieves similar quality in ~50 steps — how?
- Why predict the noise ε rather than x_0 directly? The paper shows both work,
  but ε prediction gives better sample quality. Intuitively why?
- Score matching — the connection between diffusion and score-based generative
  models (Song & Ermon 2019) is something I want to understand properly.
  The score is the gradient of the log density: ∇_x log p(x).

## Next Steps

- [ ] Read DDIM (Song et al. 2020) — deterministic sampling in fewer steps
- [ ] Read the score matching connection — Song & Ermon 2019
- [ ] Implement a tiny diffusion model on MNIST in `04-neural-networks/generative-models/`
