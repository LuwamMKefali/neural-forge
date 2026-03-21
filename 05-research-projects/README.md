# 05 · Research Projects

Extended research work — each project has a proper write-up.

## Projects

### Bias Auditing of Emotion Recognition for Neurodivergent Populations
Systematic failure analysis of DeepFace and EmotiEffLib. Accuracy on autism subsets degraded from **73% → 37%**, ECE spiked 20x. Grad-CAM and LIME identified texture-based shortcuts. Fairness Auditing Checklist aligned to the EU AI Act.
→ [IEEE Publication](#) | [Project folder](./bias-auditing)

### BioMistral-7B + VoRA: Multimodal Vision-Language Model
Extended BioMistral-7B via VoRA (Vision-as-LoRA) adapters. Full training infrastructure from scratch: custom data loading, gradient checkpointing, VQA-RAD evaluation.
→ [Project folder](./biomistral-vora)

### Celestial Object Orbit Prediction
LSTM + Transformer pipelines for multi-satellite orbital forecasting on NASA/JPL TLE data. LSTM: **~24 km MAE** at 24h horizon vs SGP4 physics baseline.
→ [Project folder](./orbital-prediction)

### Book Instance Detection + Pet Classification
Two-pipeline CV project. Classical: CLAHE + SIFT + FLANN + RANSAC → **61/60 instances detected**. Deep learning: PetResNet_v1 with 7-variable ablation study + ResNet-18 transfer learning → **89.49%**.
→ [Project folder](./book-detection-pet-classification)

### Hybrid Cognitive Robotic System
Sense-Plan-Act in ROS2/Gazebo: symbolic HTN planner + fuzzy reactive behaviours. OWL knowledge graph with owlrl. Failure detection and recovery.
→ [Project folder](./cognitive-robotics)
