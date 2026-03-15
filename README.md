# 🧠 neural-forge

> *Forging intelligence from first principles — theory, research, and production.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)  
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?logo=huggingface&logoColor=black)](https://huggingface.co)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-Tracked-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai)   
[![IEEE](https://img.shields.io/badge/IEEE-Published-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org)
[![Status](https://img.shields.io/badge/Status-Active_Research-4ADE80)](.)  

---

## Hi, I'm Luwam Major Kefali

**ML Engineer & Researcher · MSc AI @ University of Bologna**

I build deep learning systems end-to-end — from mathematical foundations to production infrastructure. My research sits at the intersection of **multimodal AI**, **fairness and bias auditing**, and **NLP for low-resource languages**.

IEEE-published author. Built VLMs, RAG pipelines, cognitive robotics systems, and ML evaluation frameworks. Currently deepening my theoretical foundations while extending my research towards top industry roles and PhD programmes.

📧 [luwammajor5@mail.com](mailto:luwammajor5@mail.com) · [LinkedIn](https://linkedin.com/in/luwam-major) · [Kaggle](https://kaggle.com/luwammajor) · [Website](#)

---

## 🗂️ What's In This Repository

| Folder | Contents |
|--------|----------|
| [`01-theory-foundations/`](./01-theory-foundations) | Backprop from scratch · Autograd · Math derivations · First-principles implementations |
| [`02-nlp-depth/`](./02-nlp-depth) | Attention from scratch · Fine-tuning · RAG system · LoRA derivation |
| [`03-computer-vision/`](./03-computer-vision) | CNN theory · ViT · Object detection · SimCLR reproduction |
| [`04-neural-networks/`](./04-neural-networks) | Optimisers from scratch · Custom layers · Generative models |
| [`05-research-projects/`](./05-research-projects) | Extended research: bias auditing, orbital prediction, multimodal VLM, CLIP |
| [`06-papers/`](./06-papers) | Annotated reading notes — one file per paper, my questions and insights |
| [`07-production-ml/`](./07-production-ml) | Dockerised projects · CI/CD · MLflow · System design |
| [`08-experiments/`](./08-experiments) | W&B logs · Ablation tables · Reproducibility records |

---

## 🔬 Research Projects

### Bias Auditing of Emotion Recognition for Neurodivergent Populations
Systematic failure analysis of DeepFace and EmotiEffLib across four datasets. Accuracy on autism subsets degraded from **73% → 37%**, ECE spiked 20x. Grad-CAM and LIME identified texture-based shortcuts. Produced a Fairness Auditing Checklist aligned to the **EU AI Act**.

→ [Project](./05-research-projects/bias-auditing) · [IEEE Publication](#)

### BioMistral-7B + VoRA: Multimodal Vision-Language Model
Extended BioMistral-7B to process image-text pairs via VoRA (Vision-as-LoRA) adapters. Full training infrastructure from scratch: custom data loading, gradient checkpointing, evaluation on VQA-RAD.

→ [Project](./05-research-projects/biomistral-vora)

### Celestial Object Orbit Prediction
LSTM and Transformer pipelines for multi-satellite orbital forecasting on NASA/JPL TLE data. LSTM achieved **~24 km MAE** at 24-hour horizon, benchmarked against SGP4 physics-based baseline.

→ [Project](./05-research-projects/orbital-prediction)

### Book Instance Detection + Pet Classification
Two-pipeline computer vision project. **Classical:** CLAHE + bilateral filter preprocessing, SIFT feature extraction, FLANN matching with Lowe's ratio test, RANSAC homography — detected **61 instances against a ground truth of 60**. **Deep learning:** custom PetResNet_v1 (~2.8M params) with Mixup augmentation (69.58%); controlled ablation across 7 architectural variables (BatchNorm, skip connections, depth, weight decay, dropout, scheduler); ResNet-18 transfer learning up to **89.49%** across three fine-tuning regimes. All experiments tracked in W&B.

→ [Project](./05-research-projects/book-detection-pet-classification)

### Hybrid Cognitive Robotic System
Sense-Plan-Act architecture in ROS2/Gazebo: symbolic HTN planner with fuzzy reactive behaviours. OWL knowledge graph with owlrl. Failure detection and recovery for real-world uncertainty.

→ [Project](./05-research-projects/cognitive-robotics)

---

## 📚 Paper Reading Log

I read and annotate 2+ papers per week. Every paper gets a structured note: summary, key equations, my questions, and reproduction status.

→ [See all paper notes](./06-papers)

**Currently reading:** *Mixtral of Experts* · *SimCLR* · *Denoising Diffusion Probabilistic Models*

---

## 🛠️ Tech Stack

```
ML/DL          PyTorch · TensorFlow · scikit-learn · HuggingFace Transformers
NLP/LLMs       LangChain · LlamaIndex · spaCy · NLTK · Flair · RAG
Computer Vision OpenCV · Ultralytics YOLOv8 · torchvision
MLOps          W&B · MLflow · Docker · Git · PostgreSQL · MongoDB
Robotics       ROS2 · Gazebo · HTN Planning · OWL Ontologies
Languages      Python (advanced) · C++ · SQL
```

---

## 📖 Publication

**Amharic Automatic Text Summarisation: Graph-Based vs. Statistical Approaches**  
H. S. Ali\*, **L. M. Kefali\***, S. Parida, S. R. Dash · OTCON'22 · *IEEE Xplore* · May 2023

---

## 📍 90-Day Research Sprint

I am currently running an intensive 90-day programme to deepen my theoretical foundations, sharpen my research skills, and build a research-grade public portfolio.

- [x] Phase 1: Theory depth — deriving what I implement
- [ ] Phase 2: Research methods — paper reading, reproduction, extension
- [ ] Phase 3: Portfolio, visibility, applications

**Commit goal:** No exceptions

---

*Bologna, Italy · March 2026* 
