# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
**Devlin et al. 2018** · [arxiv](https://arxiv.org/abs/1810.04805)  
**Read:** March 2026 · **Reproduction status:** Not started

---

## Problem

Before BERT, language model pre-training was unidirectional — GPT reads left to right,
ELMo concatenates two separate one-directional LSTMs. Neither fully exploits bidirectional
context.

For tasks like question answering, understanding a word often requires context from
both sides. "The bank by the river" vs "the bank approved the loan" — you need both
directions to disambiguate.

The question: can you pre-train a deep bidirectional transformer?

## Key Idea

Yes — but you can't just run standard language modelling bidirectionally, because that
would let every token trivially "see itself" through the context. 

The solution is two pre-training tasks:

**1. Masked Language Modelling (MLM)**
Randomly mask 15% of input tokens. The model predicts the masked tokens using
context from both directions.

Of the masked tokens:
- 80% replaced with `[MASK]`
- 10% replaced with a random token
- 10% left unchanged

The 10/10 split prevents the model from only learning to handle `[MASK]` tokens —
it has to treat every token as potentially needing prediction.

**2. Next Sentence Prediction (NSP)**
Given two sentences A and B, predict whether B actually follows A in the corpus
(50% positive, 50% random). Trains the model to understand inter-sentence relationships.

Note: later work (RoBERTa, 2019) shows NSP isn't actually helpful and removing it
improves downstream performance. But it was a reasonable hypothesis at the time.

## Architecture

Standard Transformer encoder — no decoder. Two sizes:

```
BERT-Base:  12 layers, 12 heads, d_model=768,  110M parameters
BERT-Large: 24 layers, 16 heads, d_model=1024, 340M parameters
```

Input representation:
```
[CLS] token_1 token_2 ... [SEP] token_A ... [SEP]
  ↓       ↓                  ↓
Token   Segment          Position
Embeddings Embeddings    Embeddings
         (all three summed)
```

`[CLS]` — classification token. Its final hidden state is used as the sequence
representation for classification tasks.

`[SEP]` — separator between sentence pairs.

## Fine-tuning

This is the part that made BERT so influential. After pre-training, you add a small
task-specific head and fine-tune the whole thing:

```
Classification:     [CLS] representation → linear layer → softmax
Token labelling:    each token's representation → linear layer (e.g. NER)
Question answering: predict start/end token positions in the passage
```

One model, minimal task-specific architecture, state-of-the-art on 11 NLP benchmarks.

## Connection to My Work

I've fine-tuned BERT-style models multiple times — for text classification, for the
PII anonymisation system at IAMONES, and as a baseline in my evaluation framework.

What I hadn't internalised before: the reason MLM works so well is that it forces
the model to build rich contextual representations — every token must be predictable
from its full surrounding context. That's a much harder and more informative task
than next-token prediction.

Also: the reason I could fine-tune BioMistral with LoRA rather than fully is exactly
the insight from BERT — pre-trained representations are already strong, the fine-tuning
task is just alignment, not learning from scratch.

## Questions I Still Have

- RoBERTa (Liu et al. 2019) removes NSP, trains longer, uses larger batches and
  dynamic masking — and gets significantly better results. How much of BERT's
  performance was just undertrained?
- BERT uses WordPiece tokenisation. How does this compare to BPE (used in GPT)?
  What are the tradeoffs?
- The `[CLS]` token representation is used for classification — but why would
  the representation of a special token capture the meaning of the whole sequence?
  Is this actually the best way to pool?

## Next Steps

- [ ] Fine-tune BERT on a custom classification task in `02-nlp-depth/bert-finetuning/`
- [ ] Read RoBERTa — what actually matters in BERT pre-training
- [ ] Compare `[CLS]` pooling vs mean pooling on a downstream task
